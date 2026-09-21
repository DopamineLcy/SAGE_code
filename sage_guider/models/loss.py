import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from pathlib import Path
from utils.dist_utils import all_gather_with_grad
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIHardNegativeMiner:
    """Frozen NLI inference kept outside the SAGE-guider state dict."""

    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"NLI model not found: {model_path}. See README.md for setup."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval().requires_grad_(False)
        labels = {
            str(label).lower(): int(index)
            for index, label in self.model.config.id2label.items()
        }
        self.contradiction_id = labels.get("contradiction", 0)

    @torch.no_grad()
    def contradictions(self, text_attributes, image_attributes, device):
        if self.model.device != device:
            self.model.to(device)
        inputs = self.tokenizer(
            text_attributes,
            image_attributes,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        predictions = self.model(**inputs).logits.argmax(dim=1)
        return predictions == self.contradiction_id


class SimilarityLogit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, local_tokens, temperature=1.0):
        """
        queries: [N, D] (Text features)
        local_tokens: [B, L, D] (Image Patch Tokens)
        """
        # 1. Normalize
        q = F.normalize(queries, p=2, dim=-1)      # [N, D]
        v = F.normalize(local_tokens, p=2, dim=-1) # [B, L, D]
        
        scale = 1.0 / temperature
        
        # 2. Attention-based Aggregation (Maira-2 / HalluCLIP style)
        # Compute alignment between every global text and every local image patch
        # sim_scores: [B, N, L]
        # Warning: This is extremely memory intensive if B and N are both global!
        # [B_global, N_global, L] -> e.g. [512, 2048, 256] -> 256M elements * 2/4 bytes = 0.5-1GB just for scores
        # But backward pass needs to store activations.
        sim_scores = torch.einsum('nd, bld -> bnl', q, v) * scale
        
        # Softmax over patches to find regions relevant to the text
        attn_weights = F.softmax(sim_scores, dim=-1) 
        
        # Weighted sum of image patches: [B, N, D]
        aggregated_img = torch.einsum('bnl, bld -> bnd', attn_weights, v)
        
        # 3. Final Similarity (Cosine)
        agg_norm = F.normalize(aggregated_img, p=2, dim=-1)
        
        # Logits: [B, N]
        # Dot product of Text Query vs Aggregated Image Features
        logits = (agg_norm * q.unsqueeze(0)).sum(dim=-1)
        
        return logits, sim_scores

class Loss(nn.Module):
    def __init__(self, use_vision_cls_token=True, attn_temperature=None,
                 loss_temperature=0.07, world_size=1,
                 nli_model_path=None, max_nli_pairs=2048, **kwargs):
        super().__init__()

        self.world_size = world_size
        self.use_vision_cls_token = use_vision_cls_token
        self.max_nli_pairs = max_nli_pairs
        
        # RadZeroLoss style temperatures
        self.loss_temperature = nn.Parameter(
            torch.FloatTensor([np.log(loss_temperature)])
        )
        
        if attn_temperature is not None:
            self.attn_temperature = nn.Parameter(
                torch.FloatTensor([np.log(attn_temperature)])
            )
        else:
            self.attn_temperature = None
        
        self.similarity_logit = SimilarityLogit()

        if nli_model_path is None:
            raise ValueError("nli_model_path is required for hard-negative mining")
        # The holder is intentionally not an nn.Module, so frozen NLI weights
        # are not duplicated in SAGE-guider checkpoints.
        self.nli_miner = NLIHardNegativeMiner(nli_model_path)

    def _compute_masks(self, B_global, N_global, image_presence_map, text_entity_ids, text_is_pos, text_src_indices, device,
                       text_attributes=None, image_attributes=None):
        """
        Constructs masks strictly following @models/loss_v2.py logic.
        """
        # A. Expand Image Indices
        global_img_ids = torch.arange(B_global, device=device)
        
        # B. "Is Same Instance" Mask [N_global, B_global]
        mask_same_instance = (text_src_indices.unsqueeze(1) == global_img_ids.unsqueeze(0))
        
        # C. Get Image Status for the Entity described in Text[i]
        img_presence_t = image_presence_map.t()  # [num_devices, B]
        safe_entity_ids = text_entity_ids.clone()
        safe_entity_ids[safe_entity_ids == -1] = 0
        img_ent_status = img_presence_t[safe_entity_ids] # [N, B]
        
        # D. Broadcast Text Phrasing Status
        txt_is_pos = text_is_pos.unsqueeze(1).expand(-1, B_global)
        
        # --- Logic Gates (from loss_v2.py) ---
        
        # 1. Positive Match
        is_both_positive = (img_ent_status == 1) & (txt_is_pos == 1)
        is_pos_match_yes = is_both_positive & mask_same_instance
        is_pos_match_no = (img_ent_status == 0) & (txt_is_pos == 0) & mask_same_instance
        is_pos_match = is_pos_match_yes | is_pos_match_no
        
        # 2. Ignored (Weight 0)
        valid_mask = torch.ones((N_global, B_global), device=device, dtype=torch.bool)
        
        # Exclude samples with uncertain presence labels (-100).
        valid_mask = valid_mask & (img_ent_status != -100)
        
        # Exclude positive matches originating from different images.
        is_pos_diff_loc = is_both_positive & (~mask_same_instance)
        
        # Initial valid mask filters these out
        valid_mask = valid_mask & (~is_pos_diff_loc)
        
        # --- Hard Negative Mining using NLI ---
        # If we have attributes, we can check if "diff loc" pairs are actually contradictions (Hard Negatives)
        # If contradiction, we set valid_mask = True (and pos_mask is False), so they are treated as negatives.
        if text_attributes is not None and image_attributes is not None:
            # Find candidate pairs
            # limit pairs to avoid OOM
            cand_indices = torch.nonzero(is_pos_diff_loc, as_tuple=False)
            
            # If too many pairs, sample them to keep speed
            if len(cand_indices) > self.max_nli_pairs:
                perm = torch.randperm(len(cand_indices), device=cand_indices.device)
                cand_indices = cand_indices[perm[:self.max_nli_pairs]]
            
            if len(cand_indices) > 0:
                pairs_text = []
                pairs_img = []
                valid_idx_mask = []
                
                for idx in cand_indices:
                    n_idx, b_idx = idx[0].item(), idx[1].item()
                    
                    # Safe access
                    if n_idx < len(text_attributes) and b_idx < len(image_attributes):
                        txt_attr = text_attributes[n_idx]
                        ent_id = text_entity_ids[n_idx].item()
                        
                        # Check bounds
                        if ent_id >= 0 and ent_id < len(image_attributes[b_idx]):
                            img_attr = image_attributes[b_idx][ent_id]
                            
                            if txt_attr and img_attr: # Check non-empty
                                pairs_text.append(txt_attr)
                                pairs_img.append(img_attr)
                                valid_idx_mask.append(idx)
                
                if pairs_text:
                    is_contradiction = self.nli_miner.contradictions(
                        pairs_text, pairs_img, device
                    )
                    for pair_index, is_contra in enumerate(is_contradiction.tolist()):
                        if is_contra:
                            n, b = valid_idx_mask[pair_index]
                            valid_mask[n, b] = True
        
        txt_valid = (text_entity_ids != -1).unsqueeze(1)
        valid_mask = valid_mask & txt_valid
        
        return is_pos_match, valid_mask

    def _loss(self, logits, image_presence_map, text_entity_ids, text_is_pos, text_src_indices, 
              text_attributes=None, image_attributes=None):
        """
        Computes KeyPhraseAlignmentLoss (MP-NCE) with V2 masking logic and RadZeroLoss implementation style.
        """
        # Work in [N, B] (Text x Image)
        logits_t = logits.t() # [N, B]
        N_global, B_global = logits_t.shape
        device = logits_t.device
        
        # 1. Compute Masks
        pos_mask, valid_mask = self._compute_masks(
            B_global, N_global, 
            image_presence_map, text_entity_ids, text_is_pos, text_src_indices, 
            device,
            text_attributes, image_attributes
        )
        
        # 2. Compute NCE Loss (RadZeroLoss style)
        # Pass loss_temperature here for scaling
        loss = multi_positive_masked_nce_loss(
            logits=logits_t,
            pos_mask=pos_mask,
            valid_mask=valid_mask,
            temperature=self.loss_temperature.exp()
        )
        
        return loss

    def forward(self, vision_tokens, text_features, image_presence_map, 
                text_entity_ids, text_is_pos, text_src_indices=None,
                text_attributes=None, image_attributes=None):
        """
        Args:
            vision_tokens:      [B_local, L, D]
            text_features:      [N_local, D]
            image_presence_map: [B_local, num_devices]
            text_entity_ids:    [N_local]
            text_is_pos:        [N_local]
            text_src_indices:   [N_local]
            text_attributes:    List[str] (N_local)
            image_attributes:   List[List[str]] (B_local)
        """
        device = vision_tokens.device
        
        # Remove CLS token if configured
        if not self.use_vision_cls_token:
            vision_tokens = vision_tokens[:, 1:]

        # Create source indices if not provided
        if text_src_indices is None:
            k = text_features.shape[0] // vision_tokens.shape[0]
            text_src_indices = torch.arange(vision_tokens.shape[0], device=device).repeat_interleave(k)

        # --- DDP Gather Logic ---
        if self.world_size > 1:
            # OPTIMIZATION:
            # We need to compute [B_global, N_global] logits.
            
            # 1. Gather Text
            global_text_feats = all_gather_with_grad(text_features)
            
            # Gather Metadata
            # Fix for torch.compile: using pre-allocated tensor for gather
            # We create a large tensor and gather into it
            
            # Helper for compiling-friendly gather
            def gather_tensor(local_tensor):
                # Using all_gather_with_grad for consistency and simplicity even if grad not needed, 
                # or just use dist.all_gather on a big tensor.
                # Since these are metadata (LongTensor), no grad needed.
                # Manually doing: create list -> gather -> cat is what breaks compile.
                # Instead: create Big Tensor -> dist.all_gather_into_tensor (if available) or list gather.
                # But to fix compile issue specifically with list comprehension + gather:
                # We can just use the pattern:
                gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, local_tensor)
                return torch.cat(gathered, dim=0)

            global_text_ids = gather_tensor(text_entity_ids)
            global_is_pos = gather_tensor(text_is_pos)
            
            # For src_ids, we need to adjust values after gather
            # But we can gather first, then adjust
            raw_global_src_ids = gather_tensor(text_src_indices)
            
            # Adjust Source Indices
            # raw_global_src_ids is [N_global]
            # It consists of N_local chunks. Chunk r needs += r * B_local.
            # We can do this vectorized.
            N_local = text_src_indices.shape[0]
            B_local = vision_tokens.shape[0]
            
            # Create offset vector
            # [0, 0... (N_local times), 1, 1..., 2, 2...] * B_local
            # shape [WorldSize * N_local]
            offsets = torch.arange(self.world_size, device=device).repeat_interleave(N_local) * B_local
            global_src_indices = raw_global_src_ids + offsets
            
            global_presence_map = gather_tensor(image_presence_map)
            
            # Gather Attributes (Objects)
            global_text_attributes = [None] * self.world_size
            dist.all_gather_object(global_text_attributes, text_attributes)
            # Flatten list of lists
            # global_text_attributes is [[...], [...], ...]
            global_text_attributes = [item for sublist in global_text_attributes for item in sublist]

            global_image_attributes = [None] * self.world_size
            dist.all_gather_object(global_image_attributes, image_attributes)
            global_image_attributes = [item for sublist in global_image_attributes for item in sublist]
            
            # Step 2: Compute Local Logits
            
            # RadZeroLoss logic: if attn_temperature is None, use loss_temperature
            if self.attn_temperature is not None:
                attn_temp = self.attn_temperature.exp()
            else:
                attn_temp = self.loss_temperature.exp()

            # Computes [B_local, N_global]
            local_logits, _ = self.similarity_logit(global_text_feats, vision_tokens, temperature=attn_temp)
            
            # Step 3: Gather Logits to form [B_global, N_global]
            # We use our util to allow gradients to flow back to local_logits
            global_logits = all_gather_with_grad(local_logits)
            
            # Use global_logits for loss
            logits = global_logits
            
        else:
            global_text_feats = text_features
            global_text_ids = text_entity_ids
            global_is_pos = text_is_pos
            global_src_indices = text_src_indices
            global_presence_map = image_presence_map
            global_text_attributes = text_attributes
            global_image_attributes = image_attributes
            
            if self.attn_temperature is not None:
                attn_temp = self.attn_temperature.exp()
            else:
                attn_temp = self.loss_temperature.exp()

            logits, _ = self.similarity_logit(global_text_feats, vision_tokens, temperature=attn_temp)

        # Temperature scaling is applied inside multi_positive_masked_nce_loss.
        
        loss = self._loss(
            logits=logits,
            image_presence_map=global_presence_map,
            text_entity_ids=global_text_ids,
            text_is_pos=global_is_pos,
            text_src_indices=global_src_indices,
            text_attributes=global_text_attributes,
            image_attributes=global_image_attributes
        )

        return {
            "loss": loss,
            "loss_temperature_exp": self.loss_temperature.exp(),
            "attn_temperature_exp": self.attn_temperature.exp() if self.attn_temperature is not None else torch.tensor(0.0, device=device),
            "pos_avg_logits": logits.mean(),
        }

def multi_positive_masked_nce_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
):
    """
    Computes Multi-Positive NCE loss.
    """
    # 1. Scale logits (exp)
    # Scale by 1/temperature (usually loss_temperature)
    scaled_logits = torch.exp(logits / temperature) # [N, B]
    
    # 2. Masking
    valid_scaled_logits = scaled_logits * valid_mask.float()
    pos_scaled_logits = scaled_logits * pos_mask.float()
    
    # 3. Row Loss
    row_loss = get_masked_row_loss(
        pos_scaled_logits, 
        valid_scaled_logits, 
        eps
    )
    
    # 4. Column Loss
    col_loss = get_masked_col_loss(
        pos_scaled_logits, 
        valid_scaled_logits, 
        eps
    )
    
    loss = (row_loss.mean() + col_loss.mean()) / 2
    return loss

def get_masked_row_loss(pos_scaled, valid_scaled, eps=1e-8):
    """
    Computes row loss (Text-to-Image).
    Loss = -log( sum(pos) / sum(valid) )
    Fixed for torch.compile: removes data-dependent control flow.
    """
    row_sum = valid_scaled.sum(dim=1) # [N]
    row_pos_sum = pos_scaled.sum(dim=1) # [N]
    
    # Mask of valid rows (rows with at least one positive)
    has_pos_mask = (row_pos_sum > 0).float()
    
    # Safe division
    # If row_sum is 0, then row_pos_sum is also 0. Result 0/eps = 0.
    p_row = row_pos_sum / (row_sum + eps)
    
    # Safe log
    # If p_row is 0, log(eps) -> negative large number.
    # But we will multiply by has_pos_mask=0, so it becomes 0.
    nll = -torch.log(p_row + eps)
    
    # Apply mask
    masked_nll = nll * has_pos_mask
    
    # Average over valid rows
    num_valid = has_pos_mask.sum()
    
    # Avoid div by zero if no rows are valid
    loss = masked_nll.sum() / (num_valid + eps)
    
    return loss

def get_masked_col_loss(pos_scaled, valid_scaled, eps=1e-8):
    """
    Computes column loss (Image-to-Text).
    Loss = -log( sum(pos) / sum(valid) )
    Fixed for torch.compile: removes data-dependent control flow.
    """
    col_sum = valid_scaled.sum(dim=0) # [B]
    col_pos_sum = pos_scaled.sum(dim=0) # [B]
    
    # Mask of valid cols (cols with at least one positive)
    has_pos_mask = (col_pos_sum > 0).float()
    
    p_col = col_pos_sum / (col_sum + eps)
    nll = -torch.log(p_col + eps)
    
    masked_nll = nll * has_pos_mask
    num_valid = has_pos_mask.sum()
    
    loss = masked_nll.sum() / (num_valid + eps)
    
    return loss

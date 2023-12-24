# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows,
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].
from dataclasses import dataclass
from .tokenizers import action_tokenizer
from .tokenizers import image_tokenizer
from .transformer import Transformer
from .film_efficientnet import preprocessors

from utils.misc import get_augment_transform

from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
try:
    from torchvision.transforms import v2
except:
    import torchvision.transforms as v2

from transformers import AutoModel, AutoTokenizer

@dataclass
class RT1AgentOutput:
    pred_dist: MultivariateNormal

class RT1Agent(nn.Module):
    """A transformer based actor network."""

    def __init__(
        self,
        text_enc,
        args,
        action_dim: int = 7,
        token_embedding_size: int = 512,  # RT1ImageTokenizer outputs(=context_image_tokens) has embedding dimension of token_embedding_size. This will finally be utilized in 1x1 Conv in EfficientNetEncoder class.
        language_embedding_size: int = 768,  # embedding size for language embedding and Film layers
        num_layers: int = 8,
        layer_size: int = 128,  # This corresponds to key_dim which is the size of each attention head for query, key and values.
        num_heads: int = 8,
        feed_forward_size: int = 512,  # This corresponds to d_model which is embedding dimension of each token in transformer part.
        dropout_rate: float = 0.1,
        time_sequence_length: int = 6,
        img_size: int = 300,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        use_token_learner: Optional[bool] = True,
        return_attention_scores: bool = False,
        num_encoders=1,
        using_proprioception=False,
    ):
        super().__init__()

        self._token_embedding_size = token_embedding_size
        self._language_embedding_size = language_embedding_size
        self._time_sequence_length = time_sequence_length
        self.num_encoders = num_encoders

        self.sentence_tokenizer = AutoTokenizer.from_pretrained(text_enc)
        self.sentence_encoder = AutoModel.from_pretrained(text_enc)
        self.sentence_encoder.requires_grad_(False)

        self.augment = args.augment
        self.augment_kwargs = args.augment_kwargs
        if self.augment and self.augment_kwargs:
            self.augment_kwargs['random_resized_crop']['size']  = (img_size, img_size) #TODO
            self.augment_transform = get_augment_transform(args.augment_kwargs)
        self.img_size = img_size
        self.img_resize = v2.Resize((img_size, img_size), antialias=True)
        self.img_normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=img_mean, std=img_std)
        ])

        # create tokenizers
        # self._image_tokenizers = nn.ModuleDict()
        # for idx_encoder in range(num_encoders):
        #     self._image_tokenizers[
        #         str(idx_encoder)
        #     ] = image_tokenizer.RT1ImageTokenizer(
        #         embedding_output_dim=self._token_embedding_size,
        #         language_embedding_size=self._language_embedding_size,
        #         use_token_learner=use_token_learner,
        #         num_tokens=8,
        #     )
        self._image_tokenizers = image_tokenizer.RT1ImageTokenizer(
            embedding_output_dim=self._token_embedding_size,
            language_embedding_size=self._language_embedding_size,
            use_token_learner=use_token_learner,
            num_tokens=8,
        )

        self.using_proprioception = using_proprioception
        if self.using_proprioception:
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                input_token_emb_dim=self._token_embedding_size + 1,
                return_attention_scores=return_attention_scores,
            )
        else:
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                input_token_emb_dim=self._token_embedding_size,
                return_attention_scores=return_attention_scores,
            )

        """
        following the model design in diffusion policy, we concatenate robot ee's position and orientation at the end of sequence
        """

        # Get the number of tokens
        self._tokens_per_action = action_dim
        # self._tokens_per_context_image = self._image_tokenizers[
        #     "0"
        # ].tokens_per_context_image
        self._tokens_per_context_image = self._image_tokenizers.tokens_per_context_image

        # generate loss mask and attention mask
        self._generate_masks()

        # define mappings to token embedding size
        # self._action_token_emb = nn.Linear(self._vocab_size, self._token_embedding_size)

        self._attention_scores = []
        self._use_token_learner = use_token_learner

        self.register_buffer("fixed_std", torch.eye(action_dim))

    @property
    def attention_scores(self) -> List[torch.Tensor]:
        """Return attention score. This is for debugging/visualization purpose."""
        return self._attention_scores

    def _get_action_index_for_token(self, k):
        """Returns action associated with the token at given position `k`.

        If k is not an action token then it returns -1.
        If k is part of the first action in the sequence then returns 0 etc.

        Args:
            k: an int that represents the position in the sequence.

        Returns:
            The index of the action that this position belongs to, or if this
            position is part of an image token then returns -1.
        """
        if k < 0 or k >= self._all_num_tokens:
            return -1

        n = k
        if (
            n % self._single_time_step_num_tokens < self._tokens_per_context_image
        ):  # check whether k is context_image token
            return -1
        return int(
            n / self._single_time_step_num_tokens
        )  # return which time index that k belongs to.

    # _action_tokens_mask is for loss computing. This has all indexes of action tokens in all tokens.
    # We can know which output tokens are action predictions by _action_tokens_mask - 1.
    # _default_attention_mask is modified causaul mask because we will only use observations tokens when predicting actions.
    # So we also have to mask action tokens.
    def _generate_masks(self):
        """Generate mask for action prediction loss and attention visualization."""
        # each time step = [image, action]
        self._single_time_step_num_tokens = (
            self._tokens_per_action + self._tokens_per_context_image
        )

        # full sequence = [prefix context + N x timestep + postfix context]
        self._all_num_tokens = (
            self._time_sequence_length * self._single_time_step_num_tokens
        )

        # create mask for action predition loss
        # self._action_tokens_mask has all indexes of action tokens in all tokens.
        self._action_tokens_mask = []
        for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
            for x in range(0, self._tokens_per_action, 1):
                self._action_tokens_mask.append(x + n + self._tokens_per_context_image)

        # The look ahead mask ensures causality.
        # This is a lower triangular matrix. All elements other than 0 are 1.
        # 0 means mask.
        self._default_attention_mask = torch.tril(
            torch.ones((self._all_num_tokens, self._all_num_tokens), dtype=torch.uint8)
        )

        action_mask = torch.from_numpy(
            np.ndarray(shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)
        )

        for i in range(self._all_num_tokens):
            for j in range(self._all_num_tokens):
                action_i = self._get_action_index_for_token(i)
                action_j = self._get_action_index_for_token(j)
                mask = 0
                if (
                    action_i != -1 and action_j != -1
                ):  # Check both of i and j are actions.
                    # Ignore actions of previous time steps.
                    if action_j < action_i:
                        mask = 1
                    # If we're not auto-regression, ignore action of current time step.
                    if action_j == action_i and j <= i:
                        mask = 1
                action_mask[i, j] = mask
        action_mask = action_mask.to(self._default_attention_mask.device)
        self._default_attention_mask -= action_mask

    def forward(self, prompts, obs_imgs, goal_imgs, feature_imgs):
        """Calls the transformer network.

        Args:
            observations: Observation data including image and natural language
                embedding in dict of Tensors.
        network_state: Network state data including time step, image, action
            tokens, step number in dict of Tensors.

        Returns:
            A tuple `(Detokenized output actions, network state)`.
        """

        obs_imgs = obs_imgs.to(self.fixed_std.device)
        if self.training and self.augment:
            obs_imgs = self.augment_transform(obs_imgs)
        else:
            obs_imgs = self.img_resize(obs_imgs) #TODO
        obs_imgs = self.img_normalize(obs_imgs)

        b, t = obs_imgs.shape[:2]
        # b : batch size
        # t: time_sequence_length of this model
        assert t == self._time_sequence_length

        # context_image_tokens: (b, t, num_tokens, embedding_dim)
        # action_tokens: (b, t, self._tokens_per_action)
        context_image_tokens = self._tokenize_images(obs_imgs, prompts)
        attention_mask = self._default_attention_mask

        output_tokens = self._transformer_call(
            context_image_tokens,
            attention_mask=attention_mask,
            batch_size=b,
        )

        # Gather all predicted actions for the action loss. Use fancy index to extract all predicted actions.
        predicted_action_index = torch.tensor(self._action_tokens_mask) - 1
        action_logits = output_tokens[
            :, predicted_action_index
        ]  # (bs, t*tokens_per_action, 1)
        action_logits = action_logits.view(
            b, t, self._tokens_per_action
        )  # (bs, t, self._tokens_per_action)

        dist = MultivariateNormal(action_logits, scale_tril=self.fixed_std)
        return RT1AgentOutput(
            pred_dist=dist
        )

    def _transformer_call(
        self,
        context_image_tokens: torch.Tensor,  # (b, t, num token, emb_dim)
        batch_size: int,
        attention_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_token_sequence = self._assemble_input_token_sequence(
            context_image_tokens, batch_size
        )  # [b, t*num_tokens, emb_dim]
        # run transformer
        output_tokens, self._attention_scores = self._transformer(
            input_token_sequence, attention_mask
        )  # (bs, t*num_tokens, vocab_size)
        return output_tokens

    # input_token_sequence = [context_image_tokens + action_tokens]
    def _assemble_input_token_sequence(
        self, context_image_tokens, batch_size
    ):

        b, t, _, emb_dim = context_image_tokens.shape
        action_tokens = torch.zeros(
            (b, t, self._tokens_per_action, emb_dim), dtype=context_image_tokens.dtype, device=context_image_tokens.device,
        )  # This removes autoregressively conditioning on actions becuase it did not benefit performance and slowed inference.
        # assemble token sequence
        input_token_sequence = torch.concat(
            (context_image_tokens, action_tokens), dim=2
        )
        if self.using_proprioception:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size + 1
            )
        else:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size
            )  # [b, t*num_tokens, emb_dim]
        return input_token_sequence

    # At training, we don't use network_state at all.
    # At training, this will just convert image and context into tokens.
    def _tokenize_images(self, image, prompt):
        b, input_t, _, _, _ = image.shape

        # return context from observation after check whether context is in observation.
        context = self._encoding_prompt(prompt) # [b, emb-size]
        context = context.unsqueeze(1).expand(b, input_t, -1) # [b, t, emb-size]

        # context_image_tokens = []
        # # get image tokens
        # for i in range(self.num_encoders):
        #     img = image[:, :, :, i * 256 : (i + 1) * 256, :]
        #     context_image_tokens.append(
        #         self._image_tokenizers[str(i)](img, context=context)
        #     )  # (batch, t, num_tokens, embedding_dim)
        # context_image_tokens = sum(context_image_tokens)
        context_image_tokens = self._image_tokenizers(image, context=context)
        return context_image_tokens

    # output context from observation. size: [b, t, emb-size]
    def _encoding_prompt(self, prompts):
        """Extract context from observation."""
        encoded_input = self.sentence_tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.fixed_std.device)
        encoded_output = self.sentence_encoder(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        embeddings = encoded_output[0][:, 0]
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

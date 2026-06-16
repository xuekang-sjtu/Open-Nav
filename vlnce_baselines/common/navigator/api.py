from openai import OpenAI
from PIL import Image
import torch
import numpy as np

import sys
import os

# Resolve project root for shared model/data paths (cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
# Add recognize_anything code to path
sys.path.insert(0, os.path.join(MODELS_ROOT, "recognize_anything_code"))
# Add the parent directory of SpatialBot3B so `import SpatialBot3B...` works
sys.path.insert(0, MODELS_ROOT)

from tenacity import retry, wait_random_exponential, stop_after_attempt

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from transformers import AutoConfig, AutoModelForCausalLM
from SpatialBot3B.configuration_bunny_phi import *
from SpatialBot3B.modeling_bunny_phi import *

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

from ram.models import ram
from ram import inference_ram
from ram import get_transform



class llmClient:
    def __init__(self, model_type = '', api_key=None, base_url=None):
        '''
        Initialize LLM client based on model type and API key.
        
        Args:
            model_type (str): Either "gpt" or "opensource"
            api_key (str): API key for OpenAI (if using GPT)
        '''
        api_key = self._normalize_api_key(api_key)
        # Configure based on model type
        if model_type == "gpt-4o-2024-08-06":
            self.model = model_type
            self.client = OpenAI(api_key=api_key)
            
        elif model_type == "Qwen/Qwen2-72B":
            self.model = model_type
            self.client = OpenAI(
                api_key="not-needed",  # This value doesn't matter for local deployment
                base_url=os.environ.get("OPENAI_BASE_URL", "http://0.0.0.0:23333/v1")
            )
        elif "glm" in model_type.lower():
            self.model = model_type
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://models.sjtu.edu.cn/api/v1"
            )
        else:
            self.model = model_type
            effective_base_url = os.environ.get("OPENAI_BASE_URL") or base_url or "https://models.sjtu.edu.cn/api/v1"
            self.client = OpenAI(
                api_key=api_key,
                base_url=effective_base_url
            )
        
        print(f"Initialized LLM client with model: {self.model}")

    @staticmethod
    def _normalize_api_key(api_key):
        api_key = str(api_key or "").strip()
        return api_key or "none"

    def set_model(self, model):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def gpt_infer(self, system_prompt, user_prompt, num_output=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }
        max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", "2048"))
        request_params["max_tokens"] = max_tokens

        if num_output == 1:
            chat_response = self._completion_with_backoff(**request_params)
            message = chat_response.choices[0].message
            answer = message.content if message.content is not None else (getattr(message, "reasoning", None) or "")
            return answer
        else:
            responses = []
            for _ in range(num_output):
                chat_response = self._completion_with_backoff(**request_params)
                msg = chat_response.choices[0].message
                c = msg.content if msg.content is not None else (getattr(msg, "reasoning", None) or "")
                responses.append(c)
            return responses

    
class spatialClient:
    def __init__(self, device):
        self.device = device
        self.ram_path = os.path.join(PROJECT_ROOT, "models", "recognize_anything", "pretrained", "ram_swin_large_14m.pth")
        self.spatialbot_path = os.path.join(PROJECT_ROOT, "models", "SpatialBot3B")
        self.spatialbot_device = "cuda" if torch.cuda.is_available() else "cpu"
        view_record_path = "cache_files/view_cache.json"
        try:
            from transformers import BitsAndBytesConfig
            from transformers.modeling_utils import PreTrainedModel
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            original_to = PreTrainedModel.to

            def safe_quantized_to(model, *args, **kwargs):
                return model

            PreTrainedModel.to = safe_quantized_to
            try:
                self.spatialbot_model = AutoModelForCausalLM.from_pretrained(
                    self.spatialbot_path,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=True)
            finally:
                PreTrainedModel.to = original_to
            self._use_local_spatialbot_vision_tower()
            model_device = getattr(self.spatialbot_model, "device", None)
            if model_device is not None:
                self.spatialbot_device = str(model_device)
        except ImportError:
            raise RuntimeError(
                "Open-Nav requires bitsandbytes for 4bit quantized SpatialBot3B loading on limited VRAM. "
                "Install with: pip install bitsandbytes"
            )
        except Exception as e:
            raise RuntimeError(
                "Open-Nav requires local SpatialBot3B and RAM dependencies for faithful visual perception."
            ) from e

        try:
            self.spatialbot_tokenizer = AutoTokenizer.from_pretrained(
                self.spatialbot_path,
                trust_remote_code=True,
                local_files_only=True)

            self.ram_transform = get_transform(image_size=224)
            self.ram_model = ram(pretrained=self.ram_path, image_size=224, vit='swin_l').eval().to(self.device)
        except Exception as e:
            raise RuntimeError(
                "Open-Nav requires local SpatialBot3B and RAM dependencies for faithful visual perception."
            ) from e

    def _use_local_spatialbot_vision_tower(self):
        local_siglip = os.path.join(self.spatialbot_path, "siglip-so400m-patch14-384")
        if not os.path.isfile(os.path.join(local_siglip, "model.safetensors")):
            raise RuntimeError(f"Missing local SigLIP weights: {local_siglip}")
        if hasattr(self.spatialbot_model.config, "mm_vision_tower"):
            self.spatialbot_model.config.mm_vision_tower = local_siglip
        vision_tower = self.spatialbot_model.get_vision_tower()
        if vision_tower is not None and hasattr(vision_tower, "vision_tower_name"):
            vision_tower.vision_tower_name = local_siglip
            
    def ram_img_tagging(self, image):
        ram_img = self.ram_transform(image).unsqueeze(0).to(self.device)
        img_tags = inference_ram(ram_img, self.ram_model)[0]
        return img_tags
    
    def spatialbot_description(self, image_dict, prompt):
        offset_bos = 0
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [self.spatialbot_tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(
            text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:],
            dtype=torch.long,
        ).unsqueeze(0).to(self.spatialbot_device)
        image1 = image_dict['rgb']
        image2 = image_dict['depth']
        channels = len(image2.getbands())
        if channels == 1:
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            image2 = Image.fromarray(three_channel_array, 'RGB')
        image_tensor = self.spatialbot_model.process_images(
            [image1, image2], self.spatialbot_model.config
        ).to(dtype=torch.float16, device=self.spatialbot_device)
        output_ids = self.spatialbot_model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=200,
            use_cache=True,
            repetition_penalty=1.0
        )[0]
        return self.spatialbot_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    def observe_view(self, logger, current_step, direction_idx, direction_image):
        img_tags = self.ram_img_tagging(direction_image['rgb'])
        spatial_scene_description_prompt = "What objects are in the image, and how far are these objects from the camera, calculate the result in meter."
        spatial_scene_description = self.spatialbot_description(direction_image, spatial_scene_description_prompt)
        view_observation = f"Scene Description: {spatial_scene_description} Scene Objects: {img_tags}; "
        observe_result = f"Direction {direction_idx} Direction Viewpoint ID: {direction_idx} in Step ID: {current_step} Elevation: Eye Level "  + view_observation
        return observe_result

import base64
import json
import os
from typing import Any, ClassVar, List, Optional, Set, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class MLLMImage(BaseModel):
    """Helper class for handling image URLs and local files"""
    url: str
    local: Optional[bool] = None
    # Add proper type annotation for class variable
    IMAGE_EXTENSIONS: ClassVar[Set[str]] = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"} 

    def model_post_init(self, __context) -> None:
        if not self.url.strip():
            raise ValueError("Image URL cannot be empty or whitespace")

        parsed_url = urlparse(self.url)
        if parsed_url.scheme in ['http', 'https']:
            # Handle as a remote URL
            self.local = False
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if not ext or ext not in self.IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Invalid image URL extension: '{ext}'. Supported extensions are: {', '.join(self.IMAGE_EXTENSIONS)}"
                )
            self.url = self._download_and_convert_to_base64(self.url, ext)
        elif self.is_local_path(self.url): 
            # It's a local path, check extension
            ext = os.path.splitext(self.url)[1].lower()
            if ext and ext in self.IMAGE_EXTENSIONS:
                self.local = True
                self.url = self._convert_local_to_base64(self.url)
            else:
                raise ValueError(
                    f"Local file '{self.url}' is not a recognized image type or has an invalid/missing extension. Supported extensions: {', '.join(self.IMAGE_EXTENSIONS)}"
                )
        else:
            if self.local is None: # Should be None if not processed yet
                if not self.url.startswith("data:image"):
                     raise ValueError(f"Invalid image string: '{self.url[:70]}...'. Not a downloadable URL, valid local file, or recognizable data:image URI.")
                self.local = False 

    @staticmethod
    def is_local_path(url_or_path: str) -> bool: # Renamed parameter for clarity
        # Parse the URL/path
        parsed_url = urlparse(url_or_path)

        # Check if it's a file scheme or an empty scheme indicating a local path
        if parsed_url.scheme == "file" or (parsed_url.scheme == "" and parsed_url.netloc == ""):
            # For file scheme, path is parsed_url.path. For empty scheme, path is the original string.
            path_to_check = parsed_url.path if parsed_url.scheme == "file" else url_or_path
            # Ensure the path is absolute if it's a local file reference without a scheme
            if parsed_url.scheme == "" and not os.path.isabs(path_to_check):
                # Convert to absolute so existence checks are reliable
                path_to_check = os.path.abspath(path_to_check)
            return os.path.exists(path_to_check)
        return False

    def _convert_local_to_base64(self, path: str) -> str: # Renamed for clarity
        ext = os.path.splitext(path)[1].lower()
        # Determine MIME type from extension
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg') # Default to jpeg if unknown

        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"

    def _download_and_convert_to_base64(self, url: str, ext: str) -> str:
        import requests # Import locally to avoid making requests a hard dependency if not used.
        
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg') # Default to jpeg if unknown
        
        try:
            response = requests.get(url, stream=True, timeout=30) # Standard timeout
            response.raise_for_status()  # Raise an exception for bad HTTP status codes
            
            # Convert the downloaded content to base64
            encoded_string = base64.b64encode(response.content).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"
            
        except requests.exceptions.RequestException as err: # More specific exception
            raise ValueError(f"Failed to download image from URL: {url}") from err
        except Exception as err: # Catch other potential errors during processing
            raise ValueError(f"An unexpected error occurred while processing image from URL: {url}") from err


class MLLMAudio(BaseModel):
    """Helper class for handling audio URLs and local files"""
    url: str
    local: Optional[bool] = None
    is_plain_text: bool = False
    # Add proper type annotation for class variable
    AUDIO_EXTENSIONS: ClassVar[Set[str]] = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".wma"}

    def model_post_init(self, __context) -> None:
        # Validate the URL is not empty
        if not self.url.strip():
            raise ValueError("Audio URL cannot be empty or whitespace")

        # First check if it appears to be a URL
        parsed_url = urlparse(self.url)
        if parsed_url.scheme in ['http', 'https']:
            # Handle as a remote URL
            self.local = False
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if ext and ext not in self.AUDIO_EXTENSIONS:
                raise ValueError(
                    f"Invalid audio URL extension: {ext}. Supported extensions are: {', '.join(self.AUDIO_EXTENSIONS)}"
                )
            # Download and convert URL content to base64
            self.url = self._download_and_convert_to_base64(self.url, ext)
        
        # Check directly if it's a local audio file path
        elif os.path.exists(self.url):
            ext = os.path.splitext(self.url)[1].lower()
            # print(f' Path exists : {self.url}')
            if ext in self.AUDIO_EXTENSIONS:
                self.local = True
                self.url = self._convert_to_base64(self.url)
            else:
                # It's a file but not an audio file - treat as plain text
                self.is_plain_text = True
        else:
            # It's neither a valid URL nor a local file path - treat as plain text
            self.is_plain_text = True

    def _convert_to_base64(self, path: str) -> str:
        # Get the correct MIME type based on file extension
        ext = os.path.splitext(path)[1].lower()
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.flac': 'audio/flac',
            '.wma': 'audio/x-ms-wma'
        }
        mime_type = mime_types.get(ext, 'audio/mpeg')  # Default to audio/mpeg if unknown

        with open(path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"

    def _download_and_convert_to_base64(self, url: str, ext: str) -> str:
        import requests
        import tempfile
        
        # Get the correct MIME type based on file extension
        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.flac': 'audio/flac',
            '.wma': 'audio/x-ms-wma'
        }
        mime_type = mime_types.get(ext, 'audio/mpeg')  # Default to audio/mpeg if unknown
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad responses
            
            # Convert the downloaded content to base64
            encoded_string = base64.b64encode(response.content).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"
            
        except Exception as e:
            raise ValueError(f"Failed to download audio from URL: {url}. Error: {str(e)}")


class TestCase(BaseModel):
    """
    Unified test case class that handles all types of AI model testing scenarios:
    - Basic text-based testing
    - LLM query-response testing  
    - Multimodal testing (images, audio)
    - Conversational testing
    """
    
    # Basic fields (from original TestCase)
    text: Optional[str] = None
    document: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    prompt: Optional[str] = None
    criteria: Optional[str] = None
    actual_json: Optional[dict] = None
    expected_json: Optional[dict] = None
    expected_text: Optional[str] = None
    expected_response: Optional[str] = None
    
    # LLM-specific fields (from LLMTestCase)
    query: Optional[str] = None
    response: Optional[str] = None
    context: Optional[Union[str, List[str]]] = None
    
    # Multimodal fields (from MLLMTestCase)
    image_url: Optional[Union[str, MLLMImage]] = None
    input_image_url: Optional[Union[str, MLLMImage]] = None
    output_image_url: Optional[Union[str, MLLMImage]] = None
    input_audio: Optional[Union[str, MLLMAudio]] = None
    call_type: Optional[str] = None
    
    # Conversational fields
    messages: Optional[List["TestCase"]] = Field(default=None, description="List of test cases for conversational testing")
    
    # Metadata
    test_case_type: Optional[str] = Field(default=None, description="Auto-detected or manually set test case type")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization processing for all field types"""
        
        # Handle JSON fields (from original TestCase)
        if self.actual_json is not None and isinstance(self.actual_json, str):
            self.actual_json = json.loads(self.actual_json)
        if self.expected_json is not None and isinstance(self.expected_json, str):
            self.expected_json = json.loads(self.expected_json)
        
        # Handle image fields (from MLLMTestCase)
        if isinstance(self.image_url, str):
            self.image_url = MLLMImage(url=self.image_url).url
        elif isinstance(self.image_url, MLLMImage):
            self.image_url = self.image_url.url

        if isinstance(self.input_image_url, str):
            self.input_image_url = MLLMImage(url=self.input_image_url).url
        elif isinstance(self.input_image_url, MLLMImage):
            self.input_image_url = self.input_image_url.url

        if isinstance(self.output_image_url, str):
            self.output_image_url = MLLMImage(url=self.output_image_url).url
        elif isinstance(self.output_image_url, MLLMImage):
            self.output_image_url = self.output_image_url.url

        # Handle audio fields (from MLLMTestCase)
        if isinstance(self.input_audio, str):
            self.input_audio = MLLMAudio(url=self.input_audio).url
        elif isinstance(self.input_audio, MLLMAudio):
            self.input_audio = self.input_audio.url
        
        # Handle conversational messages (from ConversationalTestCase)
        if self.messages is not None:
            if len(self.messages) == 0:
                raise TypeError("'messages' must not be empty")

            copied_messages = []
            for message in self.messages:
                if not isinstance(message, TestCase):
                    raise TypeError("'messages' must be a list of `TestCase`s")
                else:
                    # Extract query and response for conversational format
                    query = message.query or message.input or message.text or ""
                    response = message.response or message.output or message.expected_response or ""
                    copied_messages.append(str(query))
                    copied_messages.append(str(response))
            self.messages = copied_messages
        
        # Auto-detect test case type if not set
        if self.test_case_type is None:
            self.test_case_type = self._detect_test_case_type()
    
    def _detect_test_case_type(self) -> str:
        """Automatically detect the type of test case based on populated fields"""
        if self.messages is not None:
            return "conversational"
        elif any([self.image_url, self.input_image_url, self.output_image_url, self.input_audio]):
            return "multimodal"
        elif self.query is not None and self.response is not None:
            return "llm"
        elif any([self.input, self.output, self.text, self.prompt]):
            return "general"
        else:
            return "general"
    
    @classmethod
    def create_llm_test_case(
        cls,
        query: str,
        response: str,
        context: Optional[Union[str, List[str]]] = None,
        expected_response: Optional[str] = None,
    ) -> "TestCase":
        """Factory method to create an LLM test case"""
        return cls(
            query=query,
            response=response,
            context=context,
            expected_response=expected_response,
            test_case_type="llm"
        )
    
    @classmethod
    def create_multimodal_test_case(
        cls,
        query: Optional[str] = None,
        response: Optional[str] = None,
        image_url: Optional[Union[str, MLLMImage]] = None,
        input_audio: Optional[Union[str, MLLMAudio]] = None,
        **kwargs
    ) -> "TestCase":
        """Factory method to create a multimodal test case"""
        return cls(
            query=query,
            response=response,
            image_url=image_url,
            input_audio=input_audio,
            test_case_type="multimodal",
            **kwargs
        )
    
    @classmethod
    def create_conversational_test_case(
        cls,
        messages: List["TestCase"]
    ) -> "TestCase":
        """Factory method to create a conversational test case"""
        return cls(
            messages=messages,
            test_case_type="conversational"
        )


 
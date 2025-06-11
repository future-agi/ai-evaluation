import base64
import os
from typing import Optional, Union, ClassVar, Set
from urllib.parse import urlparse

from pydantic import BaseModel

from fi.testcases.general import TestCase


class MLLMImage(BaseModel):
    url: str
    local: Optional[bool] = None

    def model_post_init(self, __context) -> None:
        if self.local is None:
            self.local = self.is_local_path(self.url)
        if self.local:
            self.url = self._convert_to_base64(self.url)

    @staticmethod
    def is_local_path(url):
        # Parse the URL
        parsed_url = urlparse(url)

        # Check if it's a file scheme or an empty scheme with a local path
        if parsed_url.scheme == "file" or parsed_url.scheme == "":
            # Check if the path exists on the filesystem
            return os.path.exists(parsed_url.path)

        return False

    def _convert_to_base64(self, path: str) -> str:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"


class MLLMAudio(BaseModel):
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
            print(f' Path exists : {self.url}')
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


class MLLMTestCase(TestCase):
    image_url: Optional[Union[str, MLLMImage]] = None
    input_image_url: Optional[Union[str, MLLMImage]] = None
    output_image_url: Optional[Union[str, MLLMImage]] = None
    input_audio: Optional[Union[str, MLLMAudio]] = None
    call_type: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
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

        if isinstance(self.input, str):
            self.input = MLLMAudio(url=self.input).url
        elif isinstance(self.input, MLLMAudio):
            self.input = self.input.url
        
        if isinstance(self.output, str):
            self.output = MLLMAudio(url=self.output).url
        elif isinstance(self.output, MLLMAudio):
            self.output = self.output.url

        if isinstance(self.input_audio, str):
            self.input_audio = MLLMAudio(url=self.input_audio).url
        elif isinstance(self.input_audio, MLLMAudio):
            self.input_audio = self.input_audio.url

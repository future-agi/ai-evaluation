import httpx
from typing import Dict, Any, List, Optional

class APIRoutes:
    """
    Handles API interactions with the Future AGI backend.
    """
    def __init__(self, api_key: str, secret_key: str, base_url: str, timeout: float = 120.0):
        """
        Args:
            api_key: API key for authentication
            secret_key: Secret key for authentication
            base_url: Base URL of the backend API
            timeout: Request timeout in seconds (default: 120s for LLM operations)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "x-api-key": self.api_key,
            "x-secret-key": self.secret_key,
            "Content-Type": "application/json"
        }
        # Using a single client for connection pooling
        # Increased timeout for chat operations which may involve LLM calls
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout
        )

    async def close(self):
        await self.client.aclose()
    
    async def get_run_test_id_by_name(self, run_test_name: str) -> Dict[str, Any]:
        """
        GET /simulate/run-tests/get-id-by-name/{run_test_name}/
        Gets the run_test_id by run_test_name.
        """
        url = f"/simulate/run-tests/get-id-by-name/{run_test_name}/"
        response = await self.client.get(url)
        if response.is_error:
            self._handle_error(response, f"Failed to get run_test_id for name '{run_test_name}'")
        return response.json()
    
    def _handle_error(self, response: httpx.Response, operation: str) -> None:
        """
        Raises a more informative error with backend response details.
        """
        try:
            error_body = response.json()
            # Extract common error message patterns from backend
            if isinstance(error_body, dict):
                # Try to extract the most relevant error message
                error_msg_parts = []
                if "result" in error_body:
                    error_msg_parts.append(str(error_body["result"]))
                if "error" in error_body:
                    error_msg_parts.append(str(error_body["error"]))
                if "message" in error_body:
                    error_msg_parts.append(str(error_body["message"]))
                if "detail" in error_body:
                    error_msg_parts.append(str(error_body["detail"]))
                
                if error_msg_parts:
                    backend_error = " | ".join(error_msg_parts)
                else:
                    backend_error = str(error_body)
            else:
                backend_error = str(error_body)
        except Exception:
            error_body = response.text or f"<No response body (status {response.status_code})>"
            backend_error = error_body
        
        # Simple error message with just status code and backend error
        error_msg = f"{response.status_code}: {backend_error}"
        
        raise httpx.HTTPStatusError(
            error_msg,
            request=response.request,
            response=response
        )

    async def start_test_execution(self, run_test_id: str) -> Dict[str, Any]:
        """
        POST /simulate/run-tests/{run_test_id}/chat-execute/
        Starts a test execution and returns the execution ID.
        Note: The backend uses scenarios associated with the run_test_id.
        """
        url = f"/simulate/run-tests/{run_test_id}/chat-execute/"
        # Empty body - backend uses scenarios from run_test
        response = await self.client.post(url, json={})
        if response.is_error:
            self._handle_error(response, f"Failed to start test execution for run_id '{run_test_id}'")
        return response.json()

    async def fetch_execution_batch(
        self, 
        test_execution_id: str
    ) -> Dict[str, Any]:
        """
        POST /simulate/test-executions/{test_execution_id}/chat/call-executions/batch/
        Creates a batch of call execution IDs (has side effects - creates CallExecution records).
        Returns: {"call_execution_ids": [...], "has_more": bool, "batched_scenarios": [...]}
        """
        url = f"/simulate/test-executions/{test_execution_id}/chat/call-executions/batch/"
        
        response = await self.client.post(url, json={})
        if response.is_error:
            self._handle_error(response, f"Failed to fetch execution batch for test_execution_id '{test_execution_id}'")
        return response.json()

    async def send_chat_message(
        self, 
        call_execution_id: str, 
        messages: List[Dict[str, str]] = None,
        metrics: Dict[str, float | int] = None,
        initiate_chat: bool = False
    ) -> Dict[str, Any]:
        """
        POST /simulate/call-executions/{call_execution_id}/chat/send-message/
        Sends a message to a chat execution.
        """
        url = f"/simulate/call-executions/{call_execution_id}/chat/send-message/"
        
        payload = {
            "messages": messages,
            "metrics": metrics,
            "initiate_chat": initiate_chat
        }
        # Filter None values (but keep False for booleans if needed, though backend defaults to False)
        # We explicitly keep initiate_chat if it's True
        payload = {k: v for k, v in payload.items() if v is not None}
        
        response = await self.client.post(url, json=payload)
        if response.is_error:
            self._handle_error(response, f"Failed to send chat message for call_execution_id '{call_execution_id}'")
        return response.json()

    async def update_call_execution_status(
        self,
        call_execution_id: str,
        status: str,
        ended_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        PATCH /simulate/call-executions/{call_execution_id}/
        Updates the status of a call execution.
        
        Args:
            call_execution_id: The ID of the call execution to update
            status: The new status (must be a valid CallStatus choice, e.g., "FAILED", "CANCELLED")
            ended_reason: Optional reason for ending the call execution
        """
        url = f"/simulate/call-executions/{call_execution_id}/"
        payload = {"status": status}
        if ended_reason is not None:
            payload["ended_reason"] = ended_reason
        
        response = await self.client.patch(url, json=payload)
        if response.is_error:
            self._handle_error(response, f"Failed to update call execution status for call_execution_id '{call_execution_id}'")
        return response.json()


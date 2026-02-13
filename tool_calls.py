import requests
import json


def execute_tool_call(tool_info: dict) -> tuple[str, bool]:
    """
    Execute a single tool call via web API.

    Returns:
        Tuple of (response_text, is_error)
    """
    tool = tool_info.get("tool")

    if tool == "growth_rate":
        url = "http://growth-rate:5000/growth_rate"
        params = {"query": tool_info["query"]}
        if "level" in tool_info:
            params["level"] = tool_info["level"]
        if "stat" in tool_info:
            params["stat"] = tool_info["stat"]
        try:
            response = requests.get(url, params=params, timeout=30)
            response_text = response.text
            if response.status_code != 200:
                return response_text, True
            return response_text, False
        except requests.RequestException as e:
            return json.dumps({"error": f"Growth rate request failed: {str(e)}"}), True

    elif tool == "wom":
        url = "http://wom-flask:5000/api/query"
        payload = {k: v for k, v in tool_info.items() if k != "tool"}
        try:
            response = requests.post(url, json=payload, timeout=30)
            response_text = response.text

            if response.status_code != 200:
                return response_text, True

            try:
                result = json.loads(response_text)
                if result.get("error") or result.get("success") == False:
                    return response_text, True
            except json.JSONDecodeError:
                pass

            return response_text, False

        except requests.RequestException as e:
            return json.dumps({"error": f"Request failed: {str(e)}"}), True

    elif tool_info['tool'] == 'wom_complex':
        payload = {k: v for k, v in tool_info.items() if k != 'tool'}
        response = requests.post(
            'http://wom-flask:5000/api/complex',
            json=payload,
            timeout=30
        )
        if response.status_code != 200:
            return {"success": False, "error": f"API returned {response.status_code}: {response.text}"}, True
        try:
            return response.json(), False
        except requests.exceptions.JSONDecodeError:
            return {"success": False, "error": f"API returned non-JSON: {response.text[:500]}"}, True
    return json.dumps({"error": "Unknown tool"}), True


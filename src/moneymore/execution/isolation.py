"""Keep independent paper execution failures local, without bypassing gates."""
def isolated_execution_call(account_id, operation):
    try:
        result = operation()
        return result if isinstance(result, dict) else {
            "account_id": account_id, "status": "COMPLETED", "result": result,
        }
    except Exception as error:
        return {"account_id": account_id, "status": "EXECUTION_ERROR",
                "error": f"{type(error).__name__}: {error}"}

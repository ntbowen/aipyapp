from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator

class ExecResult(BaseModel):
    """Result of the execution of a block."""
    stdout: str | None = Field(default=None, description='Standard output')
    stderr: str | None = Field(default=None, description='Standard error')
    errstr: str | None = Field(default=None, description='Error string')
    traceback: str | None = Field(default=None, description='Traceback')

    def has_error(self) -> bool:
        return self.errstr or self.traceback or self.stderr

class ProcessResult(ExecResult):
    """Result of the execution of a process."""
    returncode: int | None = Field(default=None, description='Return code of the process')

    def has_error(self) -> bool:
        return self.returncode != 0 or super().has_error()

class PythonResult(ExecResult):
    """Result of the execution of a Python block."""
    states: Dict[str, Any] | None = Field(default=None, description='States of the execution')

    def has_error(self) -> bool:
        if super().has_error():
            return True
        
        try:
            success = self.states['success']    
        except:
            success = True
        return not success
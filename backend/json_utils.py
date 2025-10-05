"""
JSON serialization utilities for handling numpy/pandas data types
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Any, Dict, List, Union

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas data types to JSON serializable types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def safe_json_response(data: Dict) -> Dict:
    """
    Safely convert data to JSON serializable format
    """
    try:
        return convert_to_serializable(data)
    except Exception as e:
        print(f"JSON conversion error: {e}")
        # Return a safe fallback
        return {
            "success": False,
            "error": f"Data serialization error: {str(e)}",
            "fallback": True
        }

class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy/pandas types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)
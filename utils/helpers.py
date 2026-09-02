import json
import re


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def parse_llm_response_2(text: str, confidence: bool = False, two_stage: bool = False, severity: bool = False):
    # 1. Non-greedy extraction to find the JSON block
    # We look for the FIRST { and the LAST } to capture the whole object
    json_match = re.search(r'(\{.*\})', text, re.DOTALL | re.MULTILINE)
    if not json_match:
        # Fallback to pure regex if no braces are found at all
        return regex_fallback(text, confidence, two_stage, severity)

    json_str = json_match.group(1)

    try:
        # ATTEMPT 1: Standard JSON parse
        data = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # ATTEMPT 2: Repair literal newlines (Common in Gemma 3 / Llama 3)
            # This replaces actual line breaks inside quotes with the string '\n'
            repaired_json = re.sub(r'(?<=[:\s])"(.*?)"(?=[\s,}]|$)',
                                   lambda m: m.group(0).replace('\n', '\\n'),
                                   json_str, flags=re.DOTALL)
            data = json.loads(repaired_json)
        except:
            # ATTEMPT 3: If both fail, use the regex fallback
            return regex_fallback(text, confidence, two_stage, severity)

    # Normalize keys to uppercase for consistency
    data = {str(k).upper(): v for k, v in data.items()}

    result = {
        "REASONING": data.get("REASONING"),
        "PREDICTED_CATEGORY_ID": data.get("PREDICTED_CATEGORY_ID"),
        # "CONFIDENCE_SCORE": data.get("CONFIDENCE_SCORE")
    }
    if confidence:
        try:
            result["CONFIDENCE_SCORE"] = int(data.get("CONFIDENCE_SCORE"))
        except (ValueError, TypeError):
            result["CONFIDENCE_SCORE"] = data.get("CONFIDENCE_SCORE")
    if two_stage:
        # Normalize YES/NO or TRUE/FALSE to Boolean
        raw_bin = str(data.get("HAS_VIOLATION", "")).upper()
        result["HAS_VIOLATION"] = True if any(x in raw_bin for x in ["YES", "TRUE"]) else False
    if severity:
        try:
            result["SEVERITY_SCORE"] = int(data.get("SEVERITY_SCORE"))
        except (ValueError, TypeError):
            result["SEVERITY_SCORE"] = data.get("SEVERITY_SCORE")
    return result

def regex_fallback(text, confidence, two_stage, severity):
    """Stand-alone regex extractor if JSON parsing fails entirely."""
    cat_match = re.search(r'PREDICTED_CATEGORY_ID["\s:]+(S\d+)', text, re.I)
    # conf_match = re.search(r'CONFIDENCE_SCORE["\s:]+([\d.]+)', text, re.I)
    reason_match = re.search(r'REASONING["\s:]+"(.*?)"', text, re.DOTALL | re.I)

    res = {
        "REASONING": reason_match.group(1).strip() if reason_match else None,
        "PREDICTED_CATEGORY_ID": cat_match.group(1) if cat_match else None,
        # "CONFIDENCE_SCORE": float(conf_match.group(1)) if conf_match else None
    }
    if confidence:
        conf_match = re.search(r'CONFIDENCE_SCORE["\s:]+([\d.]+)', text, re.I)
        try:
            res["CONFIDENCE_SCORE"] = round(float(conf_match.group(1))) if conf_match else None
        except (ValueError, TypeError):
            res["CONFIDENCE_SCORE"] = conf_match.group(1) if conf_match else None
    if two_stage:
        bin_match = re.search(r'HAS_VIOLATION["\s:]+(YES|NO|TRUE|FALSE)', text, re.I)
        res["HAS_VIOLATION"] = bin_match.group(1).upper() in ["YES", "TRUE"] if bin_match else None
    if severity:
        severity_match = re.search(r'SEVERITY_SCORE["\s:]+([\d.]+)', text, re.I)
        #res["SEVERITY_SCORE"] = int(severity_match.group(1)) if severity_match else None
        try:
            res["SEVERITY_SCORE"] = round(float(severity_match.group(1))) if severity_match else None
        except (ValueError, TypeError):
            res["SEVERITY_SCORE"] = severity_match.group(1) if severity_match else None

    return res if (res["PREDICTED_CATEGORY_ID"] or res.get("HAS_VIOLATION") is not None) else None

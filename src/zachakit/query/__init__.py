# This dict records supported model names for local client and azure client.
SUPPORTED_MODEL_NAMES = {
    "local": [
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4-1106-vision-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-0125",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09"
    ],
    "azure": [
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-vision",
        "gpt-35-turbo",
        "gpt-35-turbo-16k",
        "gpt-35-turbo-instruct",
    ],
}

# This dict records known api token price for local client estimation.
# The price is calculated by dollar per k tokens.
TOKEN_PRICE = {
    "gpt-4-0125-preview": (0.01, 0.03),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4-1106-vision-preview": (0.01, 0.03),
    "gpt-4-0613": (0.03, 0.06),
    "gpt-4-32k-0613": (0.06, 0.12),
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-3.5-turbo-instruct": (0.0015, 0.0020),
}

MODEL_ALIAS = {
    "gpt-4-turbo-preview": "gpt-4-0125-preview",
    "gpt-4-vision-preview": "gpt-4-1106-vision-preview",
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-3.5-turbo": "gpt-3.5-turbo-1106",
}

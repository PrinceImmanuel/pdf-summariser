from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import os
print(os.environ.get("OPENAI_API_KEY"))

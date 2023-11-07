from modelsCollection.models import Models
import json
from pathlib import Path
import sys
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Logger/application.log"),
        logging.StreamHandler()
    ]
)


class ModelExecutor:
    logger = logging.getLogger(__name__)

    def __init__(self, base_path, config_prefix="config_"):
        self.BASE_PATH = Path(base_path)
        self.config_prefix = config_prefix

    def load_config(self, filename):
        config_path = self.BASE_PATH / f"config/{self.config_prefix}{filename}.json"
        with config_path.open('r') as json_file:
            config = json.load(json_file)
        return config

    def execute_model_function(self, config):
        try:
            models_instance = Models()
            functionName = config["model_function"]
            args = config["args"]

            # Take input for either 'openai_chat' or 'alephalpha_chat'
            if functionName.endswith('_chat'):
                question = input("\nEnter a query: ")
                args.append(question)

            stateMethod = getattr(models_instance, functionName)
            result=stateMethod(*args)
            answer=None
            if result and 'answer' in result:
                answer = result['answer']
                print(answer)
        except Exception as e:
            self.logger.error(f"Error executing model function: {str(e)}")
            raise

    def run(self, filename):
        try:
            config = self.load_config(filename)
            self.execute_model_function(config)
        except Exception as e:
            self.logger.error(f"Error in run method: {str(e)}")


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    executor = ModelExecutor(BASE_PATH)

    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py config_filename")
        sys.exit(1)

    modelfile = sys.argv[1]
    executor.run(modelfile)

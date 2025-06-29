# Model Evaluation & Visualization
def evaluate_llm(visualize="True"):
    print("TODO")
    if visualize:
        print("FANCY PLOT")

# Training Function
def train_loop():
    print("TODO:")

# Interface
class AthenaLLM():
    def __init__(self, model_type, wikipedia_only = True, auto_setup=True):  # params here are superset up _manual_setup
        print("TODO: user defines what they want")
        # be able to provides weights & hyperparams...
        if wikipedia_only:
            print("TODO")
        if auto_setup:
            self._manual_setup()

    def add_data(self, path):
        print("TODO: manually allow data to be added in addition to wikipedia")

    def _manual_setup(self):
        print("TODO: user begins setup (e.g. training, so forth...)")

    def save_weights(self):
        print("TODO: save weights/hyperparams")

    def prompt(self):
        print("TODO: user can use")

class Parameters:
    num_trials: int = 1
    num_rows: int = 30  # Set to None to process the entire dataset
    contexts: list = ['Job Applicant']  # Specify a list of context ids, e.g., ['context_1', 'context_2']
    likert_scales: list = [5]  # Specify a list of Likert scales, e.g., [3, 5, 7]
    dimensions: list = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
    temperature: float = 0
    buffer_tokens: int = 500  # Tokens reserved for model response
    batching_method: str = 'bulk'  # Options: 'bulk', 'incremental', 'none'
    use_reasoning: bool = False
    verbose: bool = True  # Set to True for detailed logging
    batch_size: int = 30 if use_reasoning else num_rows  # Maximum number of comparisons per batch

    original_context: tuple = (
        "There is a hiring manager for an organization focused on improvement and progress.", "Hiring Manager")
    context_file: str = 'data/contexts.xlsx'
    data_folder: str = 'data/cover_letters'
    results_folder: str = "results"

    @classmethod
    def display(cls):
        print("Experiment Parameters:")
        for attr, value in cls.__dict__.items():
            if not attr.startswith("__"):
                print(f"  {attr}: {value}")



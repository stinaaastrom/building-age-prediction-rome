from datasets import load_dataset

class RomeDataset:
    def __init__(self, dataset_name="Morris0401/Year-Guessr-Dataset"):
        self.dataset_name = dataset_name

    def _filter_condition(self, example):
        # Extract fields
        desc = example.get('Description', '')
        country = example.get('Country', '')

        is_rome_desc = False
        if desc and country:
            if ('Rom' in desc or 'rom' in desc) and country == 'Italy':
                is_rome_desc = True
                
        return is_rome_desc

    def get_filtered_dataset(self, split="train"):
        print(f"Loading dataset '{self.dataset_name}' split '{split}'...")
        dataset = load_dataset(self.dataset_name, split=split)
        
        print(f"Filtering dataset for Rome (split={split})...")
        filtered_dataset = dataset.filter(self._filter_condition, num_proc=4)
        
        print(f"Original size: {len(dataset)}, Filtered size: {len(filtered_dataset)}")
        return filtered_dataset

import requests
from bs4 import BeautifulSoup

# URL of the Hugging Face page

# File path
file_path = "datasets.txt"

class ExtractNewDataset:
    def __init__(self, file_path="datasets.txt"):
        self.file_path = file_path

    # Function to read the text file and convert it into a list
    def read_file_to_list(self):
        try:
            with open(self.file_path, "r") as file:
                lines = file.read().splitlines()  # Read and split by lines
            return lines
        except FileNotFoundError:
            print("File not found. Please make sure the file exists.")
            return []

    def append_items_to_file(self, new_items):
        with open(file_path, "a") as file:
            file.write("\n".join(new_items) + "\n")
            print(f"Added {len(new_items)} new items to the file.")


    def get_new_items(self, url = "https://huggingface.co/crtvai"):
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            dataset = soup.find_all("article")
            dataset_list = []
            for data in dataset:
                header = data.find("header")
                title = header['title']
                # print(title)
                dataset_list.append(title)
        print(dataset_list)
        already_exist = self.read_file_to_list()
        new_dataset_list = list(set(dataset_list) - set(already_exist))
        self.append_items_to_file(new_dataset_list)
        return new_dataset_list
        # print(new_dataset_list)
        # already_exist.append(new_dataset_list)
        # print(already_exist)
        # return dataset_list


    
if __name__=="__main__":
    extract_new_dataset = ExtractNewDataset()
    new_dataset_list = extract_new_dataset.get_new_items()
    print(new_dataset_list)

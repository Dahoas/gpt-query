class FString:
    text: str
    
    def __init__(self, text):
        self.text = text
        
    def format(self, **kwargs):
        for key, value in kwargs.items():
            if "{"+key+"}" not in self.text:
                raise ValueError(f"{key} field not found!")
            else:
                self.text = self.text.replace("{"+key+"}", str(value))
        return self.text

    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text
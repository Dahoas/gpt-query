"""
Implements a partially substitutable format string.
"""

class FString:
    text: str
    
    def __init__(self, text):
        self.text = text
        
    def format(self, **kwargs):
        for key, value in kwargs.items():
            if "{"+key+"}" in self.text:
                self.text = self.text.replace("{"+key+"}", str(value))
        return self.text

    def __str__(self):
        return self.text
    
    def __repr__(self):
        return self.text

from jinja2 import Template

class PromptTemplate:
    ''' A class to represent a template for generating prompts with variables
    Attributes:
        template (str): The template string with variables
        input_variables (list): A list of the variable names in the template
    '''
    def __init__(self, template, input_variables):
        self.template = Template(template)
        self.input_variables = input_variables
    
    def format(self, **kwargs):
        return self.template.render(**kwargs)
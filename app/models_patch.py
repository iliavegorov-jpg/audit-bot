import re

with open('app/models.py', 'r') as f:
    content = f.read()

# заменяем SectionVariants
old_class = '''class SectionVariants(BaseModel):
    variants: Union[List[str], Dict[str, Any]] = Field(
        ...,
        description="Either list of variant strings OR structured object"
    )'''

new_class = '''class SectionVariants(BaseModel):
    text: Optional[str] = None
    variants: Union[List[str], Dict[str, Any]] = Field(
        ...,
        description="Either list of variant strings OR structured object"
    )'''

content = content.replace(old_class, new_class)

with open('app/models.py', 'w') as f:
    f.write(content)

print("models.py обновлён!")

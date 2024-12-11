import xml.etree.ElementTree as ET


def parse_element(element):
    children = list(element)
    if not children:  # If no children, return the text
        return element.text.strip() if element.text else None
    return {child.tag.lower(): parse_element(child) for child in children}


def j2_to_dict(file_path):
    """Convert Jinja2-like XML content into a dictionary.

    Parameters:
    - xml_content (str): XML-like string content.

    Returns:
    - dict: A nested dictionary representing the XML structure.
    """
    with open(file_path) as op:
        xml_content = op.read()

    # Parse the XML content
    root = ET.fromstring(xml_content)
    return {root.tag.lower(): parse_element(root)}['root']

import re
import random

def parse_evaluation(text):
    score_str = 'E'
    # Regex pattern to match {0} to {10}
    #pattern = r"(?<!\d)(\d+|X)(?!\d)"
    pattern = r"\b\d+\b(?=\.\s|\s|$)|\bnot available\b"
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last match
        score = matches[-1]

        if score == 'not available':
            score_str = 'X'
        else:
            try:
                int_score = round(float(score))
                if int_score < 0 or int_score > 10:
                    score_str = "not in range"
                else:
                    score_str = str(int_score)
            except:
                score_str = 'E'

    # we just return the whole text as the assessment
    assessment = text

    return assessment, score_str


def text_to_html(text, border_radius = 0, padding = 10, margin = 0,
                 color = '#FFF', background_color = '#28a745',
                 strong = False):
    # Validate and set default values if necessary
    if isinstance(border_radius, (int, float)):
        border_radius_style = f"{border_radius}px"
    elif isinstance(border_radius, tuple) and len(border_radius) == 4 and all(isinstance(n, (int, float)) for n in border_radius):
        border_radius_style = ' '.join(f"{n}px" for n in border_radius)
    else:
        border_radius_style = "0px"
    if isinstance(margin, (int, float)):
        margin_style = f"{margin}px"
    elif isinstance(margin, tuple) and len(margin) == 4 and all(isinstance(n, (int, float)) for n in margin):
        margin_style = ' '.join(f"{n}px" for n in margin)
    else:
        margin_style = "0px"
    padding = 10 if not isinstance(padding, (int, float)) else padding
    color = '#FFF' if not isinstance(color, str) else color
    background_color = '#28a745' if not isinstance(background_color, str) else background_color

    # Apply the <strong> tag
    if strong and isinstance(strong, bool):
        formatted_text = f"<strong>{text}</strong>"
    else:
        formatted_text = text

    # Build the css using the provided parameters
    css = (f"border-radius: {border_radius_style}; "
           f"padding: {padding}px; "
           f"margin: {margin_style}; "
           f"color: {color}; "
           f"background-color: {background_color};")

    return f"""
            <div style="{css}">
                {formatted_text}
            </div>
            """

def get_color(score):
    color_best = '#37804b'
    color_worst = '#875050'

    def hex_to_rgb(hex_color):
        # Convert a hex color string to an RGB tuple
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    try:
        score_int = int(score)

        if not 0 <= score_int <= 10:
            raise ValueError  # Ensure score is within the range 0 to 10

        # Calculate the ratio of the interpolation
        ratio = score_int / 10

        # Convert hex colors to RGB
        rgb_best = hex_to_rgb(color_best)
        rgb_worst = hex_to_rgb(color_worst)

        # Interpolate between the two colors
        mixed_rgb = tuple(round(ratio * b + (1 - ratio) * w) for b, w in zip(rgb_best, rgb_worst))

        # Convert back to RGB color string
        return f"rgb({mixed_rgb[0]}, {mixed_rgb[1]}, {mixed_rgb[2]})"

    except ValueError:
        return "rgb(0, 0, 0)"  # Black color for invalid scores

def calculate_average(values):
    int_values = []
    count_x = 0
    count_e = 0
    count_int_scores = 0

    for value in values:
        if value == 'X':
            count_x += 1
        elif value == 'E':
            count_e += 1
        else:
            try:
                # Convert to float and round to integer, count as an integer score
                int_value = round(float(value))
                int_values.append(int_value)
                count_int_scores += 1
            except ValueError:
                # Handle the case where the value is not a number
                print(f"Value '{value}' is not a valid number and will be ignored.")

    # Calculate the average if the list of integers is not empty
    average = sum(int_values) / len(int_values) if int_values else None

    return average, count_x, count_e, count_int_scores

def get_random_score():
    # Define the list of possible scores
    possible_scores = list(range(11)) + ['X', 'E']
    
    # Randomly select and return one score
    return random.choice(possible_scores)


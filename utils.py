from PIL import Image, ImageDraw
import io
import base64

def drawrect(x, y,xstr,ystr):
    d_rect = True

    if x * y <= 10000:
        recWidth = x * 5
        recHeight = y * 5
    else:
        recWidth = x
        recHeight = y

    h = xstr 
    w = ystr 
    recWidth=int(recWidth)
    recHeight=int(recHeight)
    # Create a new image and draw object
    img = Image.new('RGB', (int(recWidth) + 80, int(recHeight) + 80), color='white')

    draw = ImageDraw.Draw(img)

    # Draw rectangle with thicker stroke
    draw.rectangle([(10, 10), (int(recWidth) + 10, int(recHeight) + 10)], width=2, outline='black', fill=None)
       # draw.rectangle((x1, y1, x2, y2), outline='black', width=3)

    # Calculate text position and draw the dimensions
    h_text_size = draw.textsize(h)
    w_text_size = draw.textsize(w)
    h_pos = ((recWidth + 30 - h_text_size[0]) // 2, 12)
    w_pos = (recWidth + 9 - w_text_size[0], (recHeight + 20 - w_text_size[1]) // 2)
    draw.text(h_pos, h, fill='black')
    draw.text(w_pos, w, fill='black')

    # Convert image to base64 string and return
    buffered = io.BytesIO()
    img.show()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

import math
import io
def draw_parallelogram(width, height,wstr,hstr, angle_degrees=60):
    # Convert angle to radians
    angle = angle_degrees * math.pi / 180.0

    # Calculate coordinates of parallelogram vertices
    x1 = 0
    y1 = 0
    x2 = width
    y2 = 0
    x3 = width + height * math.cos(angle)
    y3 = height * math.sin(angle)
    x4 = height * math.cos(angle)
    y4 = height * math.sin(angle)

    # Create a new image and draw object
    img = Image.new('RGB', (max(int(x3)+20, int(x4)+20), max(int(y3)+20, int(y4)+20)), color='white')
    draw = ImageDraw.Draw(img)

    # Draw parallelogram
    draw.polygon([(x1+10, y1+10), (x2+10, y2+10), (x3+10, y3+10), (x4+10, y4+10)], outline='black', width=2)

    # Draw axes with labels
    draw.line([(x1+10, y1+10), (x2+10, y2+10)], fill='black', width=2)
    draw.line([(x2+10, y2+10), (x3+10, y3+10)], fill='black', width=2)
    draw.text((x2/2+10, 0), wstr, fill='black')
    draw.line([(x1+10, y1+10), (x4+10, y4+10)], fill='black', width=2)
    draw.line([(x4+10, y4+10), (x3+10, y3+10)], fill='black', width=2)
    draw.text((x4/2-5, y4/2+10), hstr, fill='black')

    # Convert image to base64 string and return
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


 
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
def draw_circle(radius,a):
    # Clear the current figure
    plt.clf()
    
    # Create a new figure with larger size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw circle with thicker stroke
    circle = Circle((radius + 40, radius + 40), radius, edgecolor='black', facecolor='none', linewidth=4)
    ax.add_patch(circle)
    
    # Draw the radius line
    center = (radius + 40, radius + 40)
    end = (center[0] + radius, center[1])
    ax.plot([center[0], end[0]], [center[1], end[1]], color='black', linewidth=4)

    # Set axis limits and turn off the axis ticks
    ax.set_xlim([0, 2*radius+80])
    ax.set_ylim([0, 2*radius+80])


    # Draw the radius value
    text = f"Radius = {a}"
    ax.text((2*radius+80)/2, 10, text, ha='center')

    # Save figure to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    # Convert buffer to base64 string and return
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


import matplotlib.pyplot as plt

def draw_rhombus(diagonal1, diagonal2,dstr1, dstr2):
    # Calculate the half-lengths of each diagonal
    half_d1 = diagonal1 / 2
    half_d2 = diagonal2 / 2

    # Create a new plot and set the axis limits
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(diagonal1, diagonal2) + 20)
    ax.set_ylim(0, max(diagonal1, diagonal2) + 20)

    # Calculate the coordinates of the four vertices
    x1 = 0
    y1 = half_d2
    x2 = half_d1
    y2 = 0
    x3 = diagonal1
    y3 = half_d2
    x4 = half_d1
    y4 = diagonal2

    # Draw the rhombus and the line from the center to one midpoint
    ax.plot([x1+10, x2+10, x3+10, x4+10, x1+10], [y1+10, y2+10, y3+10, y4+10, y1+10], color='black', linewidth=2)
    ax.plot([half_d1+10, x2+10], [half_d2+10, y2+10], color='black', linewidth=2)

    # Add labels for the diagonals
    ax.text(half_d1/2+10, half_d2+5, dstr2, ha='center', va='bottom')
    ax.text(half_d1+5, diagonal2/2+10, dstr1, ha='left', va='center')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    # Convert buffer to base64 string and return
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def metric_conversion(l):
    available_units = ('Millimetre', 'Centimetre', 'Decimetre','Metre', 'Kilometre')
    conversions = (1, 10,100, 1000, 1e6)
    index = 0
    unit = l[1]
    num2=l[0]
    for i in range (0, len(available_units)):
        if available_units[i] == str(l[1]):
            num_in_mm = l[0] * conversions[i]
            break
    for j in range (0, len(available_units)):
        if ((num_in_mm / conversions[j])>9.99) and ((num_in_mm / conversions[j])<1000.01) :
            unit=available_units[j]
            num2 = num_in_mm / conversions[j]
            break
    return [num2,l[0],l[1]]

import re

def replace_numbers_with_digits_en(paragraph):
    number_mapping = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'eleven': '11',
        'twelve': '12',
        'thirteen': '13',
        'fourteen': '14',
        'fifteen': '15',
        'sixteen': '16',
        'seventeen': '17',
        'eighteen': '18',
        'nineteen': '19',
        'twenty': '20',
        'thirty': '30',
        'forty': '40',
        'fifty': '50',
        'sixty': '60',
        'seventy': '70',
        'eighty': '80',
        'ninety': '90'
    }

    tens_mapping = {
        'twenty': '2',
        'thirty': '3',
        'forty': '4',
        'fifty': '5',
        'sixty': '6',
        'seventy': '7',
        'eighty': '8',
        'ninety': '9'
    }

    pattern = '|'.join(r'\b{}\b'.format(re.escape(k)) for k in number_mapping.keys())
    # Compile a regex pattern to match whole words of the numbers

    def replace(match):
        matched_word = match.group(0)
        if matched_word in tens_mapping:
            return tens_mapping[matched_word] + '0'
        else:
            return number_mapping[matched_word]

    # Define a callback function to replace the matched words with their corresponding digits

    result = re.sub(pattern, replace, paragraph, flags=re.IGNORECASE)
    # Use the 're.sub' function to replace the matched words with digits, ignoring the case

    return result



import re

def replace_numbers_with_digits_ar(paragraph):
    number_mapping = {
        'صفر': '0',
        'واحد': '1',
        'اثنان': '2',
        'ثلاثة': '3',
        'أربعة': '4',
        'خمسة': '5',
        'ستة': '6',
        'سبعة': '7',
        'ثمانية': '8',
        'تسعة': '9',
        'عشرة': '10',
        'أحد عشر': '11',
        'اثنا عشر': '12',
        'ثلاثة عشر': '13',
        'أربعة عشر': '14',
        'خمسة عشر': '15',
        'ستة عشر': '16',
        'سبعة عشر': '17',
        'ثمانية عشر': '18',
        'تسعة عشر': '19',
        'عشرون': '20',
        'واحد وعشرون': '21',
        'اثنان وعشرون': '22',
        'ثلاثة وعشرون': '23',
        'أربعة وعشرون': '24',
        'خمسة وعشرون': '25',
        'ستة وعشرون': '26',
        'سبعة وعشرون': '27',
        'ثمانية وعشرون': '28',
        'تسعة وعشرون': '29',
        'ثلاثين': '30',
        'واحد وثلاثين': '31',
        'اثنان وثلاثين': '32',
        'ثلاثة وثلاثين': '33',
        'أربعة وثلاثين': '34',
        'خمسة وثلاثين': '35',
        'ستة وثلاثين': '36',
        'سبعة وثلاثين': '37',
        'ثمانية وثلاثين': '38',
        'تسعة وثلاثين': '39',
        'أربعين': '40',
        'واحد وأربعين': '41',
        'اثنان وأربعين': '42',
        'ثلاثة وأربعين': '43',
        'أربعة وأربعين': '44',
        'خمسة وأربعين': '45',
        'ستة وأربعين': '46',
        'سبعة وأربعين': '47',
        'ثمانية وأربعين': '48',
        'تسعة وأربعين': '49',
        'خمسين': '50',
        'اثنان وخمسين': '52',
        'ثلاثة وخمسين': '53',
        'أربعة وخمسين': '54',
        'خمسة وخمسين': '55',
        'ستة وخمسين': '56',
        'سبعة وخمسين': '57',
        'ثمانية وخمسين': '58',
        'تسعة وخمسين': '59',
        'ستين': '60',
        'واحد وستين': '61',
        'اثنان وستين': '62',
        'ثلاثة وستين': '63',
        'أربعة وستين': '64',
        'خمسة وستين': '65',
        'ستة وستين': '66',
        'سبعة وستين': '67',
        'ثمانية وستين': '68',
        'تسعة وستين': '69',
        'سبعين': '70',
        'واحد وسبعين': '71',
        'اثنان وسبعين': '72',
        'ثلاثة وسبعين': '73',
        'أربعة وسبعين': '74',
        'خمسة وسبعين': '75',
        'ستة وسبعين': '76',
        'سبعة وسبعين': '77',
        'ثمانية وسبعين': '78',
        'تسعة وسبعين': '79',
        'ثمانين': '80',
        'واحد وثمانين': '81',
        'اثنان وثمانين': '82',
        'ثلاثة وثمانين': '83',
        'أربعة وثمانين': '84',
        'خمسة وثمانين': '85',
        'ستة وثمانين': '86',
        'سبعة وثمانين': '87',
        'ثمانية وثمانين': '88',
        'تسعة وثمانين': '89',
        'تسعين': '90',
        'واحد وتسعين': '91',
        'اثنان وتسعين': '92',
        'ثلاثة وتسعين': '93',
        'أربعة وتسعين': '94',
        'خمسة وتسعين': '95',
        'ستة وتسعين': '96',
        'سبعة وتسعين': '97',
        'ثمانية وتسعين': '98',
        'تسعة وتسعين': '99'
    }
    pattern = '|'.join(r'\b{}\b'.format(re.escape(k)) for k in number_mapping.keys())
    # Compile a regex pattern to match whole words of the numbers

    def replace(match):
        return number_mapping[match.group(0)]

    # Define a callback function to replace the matched words with their corresponding digits

    result = re.sub(pattern, replace, paragraph)
    # Use the 're.sub' function to replace the matched words with digits

    return result
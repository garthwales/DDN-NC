import os, cv2, random
import numpy as np
from glob import glob
from matplotlib import cm
from matplotlib.image import imsave

# GENERATES num_images TEXTURE COLOUR DATASET IMAGES
def texture_colour(path, num_images):
    print(f'begin creating texture colour dataset')

    dir = path+ 'tc/'
    tc_img = dir+'img/'
    masksT = dir+'maskT/'
    masksC = dir+'maskC/'

    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + ' has been made')

    if not os.path.exists(tc_img):
        os.makedirs(tc_img)
        print(tc_img + ' has been made')

    if not os.path.exists(masksT):
        os.makedirs(masksT)
        print(masksT + ' has been made')

    if not os.path.exists(masksC):
        os.makedirs(masksC)
        print(masksC + ' has been made')

    folder_path = './data/textures/Normalized Brodatz'
    extension = '.tif'
    imgs = glob(os.path.join(folder_path,'*'+extension))
    print(f'number of textures available: {len(imgs)}')

    # count number of images in path...
    for i in range(num_images):
        # could use this way instead? and PIL Image, but this works fine!
        # im = Image.fromarray(cm.gist_earth(myarray, bytes=True))
        image, colour_mask, texture_mask = make_texture_colour_image(imgs, cmaps)

        image_name = tc_img+'img'+str(i)+'.png' # NOTE: was originally jpg but switched to pngs..?
        imsave(image_name, image)
        mask_colour_name = masksC+'img'+str(i)+'.png' # NOTE: was originally gif but switched to pngs..?
        imsave(mask_colour_name, colour_mask, cmap='Greys')
        mask_texture_name = masksT+'img'+str(i)+'.png' # NOTE: was originally gif but switched to pngs..?
        imsave(mask_texture_name, texture_mask, cmap='Greys')

    print(f'Generated {i+1} images')

def bresenhams(array,x1,y1,x2,y2, fill_value=255):
    # from https://github.com/encukou/bresenham/blob/master/bresenham.py
    dx = x2 - x1
    dy = y2 - y1

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0


    for x in range(dx + 1):
        a,b = x1 + x*xx + y*yx, y1 + x*xy + y*yy

        array[:a,b] = fill_value
        array[a:,b] = 255-fill_value

        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

# from https://stackoverflow.com/a/50692782
def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos+w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w-max(pos+w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]

def generate_colour_mask(img_size):
    """ step 1 tc dataset """
    ## GENERATE BASIC BINARY IMAGE
    x,y = 0,0
    coords = [(0,0)] # TODO: make it loop over the list itself instead? not neccessary
    while x < img_size[0]-1 or y < img_size[1]-1:
        x = min(x + random.randint(10,40), img_size[0]-1)
        y = min(y + random.randint(10,40), img_size[1]-1)
        coords.append((x,y))

    new_image = np.zeros(img_size, np.uint8)

    value = random.randint(40,255)
    for c1, c2 in zip(coords, coords[1:]):
        # draw a line between each of the randomly generated coordinates
        bresenhams(new_image,*c1,*c2, fill_value=value)

    return new_image, value

def generate_texture_mask(imgs, w, h):
    """ step 2 tc dataset"""
    ## CROP THE TEXTURE TO PLACE INTO BASIC BINARY MASK
    random_image = cv2.imread(random.choice(imgs), cv2.COLOR_BGR2GRAY)
    max_x, max_y = random_image.shape 
    x,y = random.randint(0, max_x-w), random.randint(0, max_y-h)
    crop = random_image[y:y+h, x:x+w] # crops the image into a w,h portion
    return crop

def generate_split_texture(new_image, crop, value, img_size, w,h):
    """ step 3 tc dataset """
    new_x, new_y = random.randint(0, img_size[0]-(w)), random.randint(0, img_size[1]-(h))

    base_textureA = np.zeros(img_size, np.uint8)
    paste(base_textureA, crop, (new_x, new_y))
    base_textureB = np.zeros(img_size, np.uint8)
    paste(base_textureB, crop, (new_x, new_y))

    if value > 255//2:
        a = np.transpose(np.nonzero(new_image==value))
        b = np.transpose(np.nonzero(new_image<value))
    else:
        a = np.transpose(np.nonzero(new_image==value))
        b = np.transpose(np.nonzero(new_image>value))

    for coord in b:
            x,y = coord
            base_textureA[x][y] = 0

    for coord in a:
        x,y = coord
        base_textureB[x][y] = 0


    texture_mask = np.zeros(img_size, np.uint8)
    texture_mask[new_x:new_x+w, new_y:new_y+h] = 255

    return base_textureA, base_textureB, b, texture_mask

def generate_combo_texture(new_image, value, base_textureA, base_textureB):
    """ step 4 tc dataset   """
    new_imageA, new_imageB = new_image.copy(), new_image.copy()
    new_imageA[new_imageA <= value] = value # set to 0 to see the actual line
    new_imageB[new_imageB > value] = 255-value # set to 0 to see the actual line

    # get coords from each texture, place the values of those textures into their new_image
    textureCoordsA = np.transpose(np.nonzero(base_textureA>0))
    textureCoordsB = np.transpose(np.nonzero(base_textureB>0))

    # texture_change = random.randint(10,20)
    texture_change = 0
    for coord in textureCoordsA:
        x,y = coord
        new_imageA[x][y] = min(base_textureA[x][y]+texture_change,255)

    for coord in textureCoordsB:
        x,y = coord
        new_imageB[x][y] = min(base_textureB[x][y]+texture_change, 255)

    return new_imageA, new_imageB

def generate_cmapped_halves(cmaps, new_imageA, new_imageB):
    """ step  5 tc dataset """
    if len(new_imageA.shape) < 3:
        a_cmap, b_cmap = np.random.choice(cmaps, 2, replace=False)

        a_cmap_cm = cm.get_cmap(a_cmap)
        new_imageA = a_cmap_cm(new_imageA)

        b_cmap_cm = cm.get_cmap(b_cmap)
        new_imageB = b_cmap_cm(new_imageB)

    return new_imageA, new_imageB

def generate_combined_tc(b, new_imageA, new_imageB):
    """ step 6 tc dataset """
    for coord in b:
        x,y = coord
        new_imageA[x][y] = new_imageB[x][y]

    return new_imageA

# MAKES A SINGLE TC IMAGE
def make_texture_colour_image(imgs, cmaps):
    """ createa a signle tc dataset image """
    w,h = random.randint(40,100),random.randint(40,100)
    img_size = (100,100)

    colour_mask, value = generate_colour_mask(img_size)
    texture =  generate_texture_mask(imgs, w, h)
    
    base_textureA, base_textureB, b, texture_mask = generate_split_texture(colour_mask, texture, value, img_size, w, h)
    new_imageA, new_imageB = generate_combo_texture(colour_mask, value, base_textureA, base_textureB)

    new_imageA, new_imageB = generate_cmapped_halves(cmaps, new_imageA, new_imageB)

    final_image = generate_combined_tc(b, new_imageA, new_imageB)
    
    # TODO: fix texture mask for output
    return final_image, colour_mask, texture_mask

def main():
    path = 'data/'
    num_images = 500

    texture_colour(path, num_images)

if __name__ == '__main__':
    main() 
    
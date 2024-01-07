from torchvision import transforms
import os
import random
import torch
import os, numpy, torch, csv, re, os, zipfile
from collections import OrderedDict
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from urllib.request import urlopen


broden_concept_spaces = {"color":["white-c","black-c","yellow-c","green-c","red-c","blue-c","pink-c","orange-c","brown-c","grey-c"],
                            "texture":["dotted", "striped", "zigzagged", "flecked", "smeared", "knitted",   "fibrous", "veined","bumpy","banded", "perforated", "woven", "pitted","porous", "meshed", "crosshatched", "blotchy","sprinkled", "polka-dotted", "marbled", "stained", "grid", "gauzy", "interlaced", "frilly", "spiralled", "swirly", "cracked", "studded", "matted", "potholed", "scaly", "stratified", "braided", "lined", "wrinkled", "paisley", "waffled", "freckled", "honeycombed", "lacelike", "chequered", "crystalline", "bubbly", "grooved", "pleated", "cobwebbed"][:8],
                            # "material":["wood", "painted", "fabric", "glass", "metal", "tile", "carpet", "plastic-opaque", "granite", "ceramic", "paper", "food", "leather", "plastic-clear", "laminate", "mirror", "brick", "wallpaper", "hair", "skin", "cardboard", "wicker", "concrete", "rock", "fur", "rubber", "foliage", "linoleum", "sky", "water", "blackboard", "fire"][:10],
                            # "part":["head", "leg", "torso", "arm", "eye", "windowpane", "ear", "nose", "neck", "door", "hand", "wheel", "mouth", "hair", "foot", "eyebrow", "handle", "tail", "muzzle", "headlight", "mirror", "drawer", "back", "shade", "paw", "column", "body", "knob", "base", "headboard", "faucet", "license plate", "top", "screen", "seat", "seat cushion", "wing", "roof", "railing", "pane", "rim", "taillight", "back pillow", "windshield", "balcony", "door frame", "beak", "seat base", "inside arm", "cap", "stern", "frame", "outside arm", "footboard", "plant", "pot", "apron", "handle bar", "shop window", "engine", "saddle", "chain wheel", "side", "lid", "chimney", "bumper", "coach", "skirt", "stile", "double door", "button panel", "crosswalk", "keyboard", "stove", "shelf", "monitor", "oven", "hoof", "front", "canopy", "water tank", "bowl", "backplate", "shutter", "mouse", "face", "arm panel", "muntin", "sash", "computer case", "metal shutter", "blade", "windows", "stretcher", "lower sash", "dormer", "pipe", "tap", "upper sash", "entrance", "corner pocket", "spindle", "bed", "diffusor", "side rail", "dome", "bannister", "beam", "casing", "pedestal", "sill", "garage door", "shaft", "head roof", "capital", "stairs", "rocking chair", "trunk", "vent", "fuselage", "clouds", "step", "earmuffs", "loudspeaker", "riser", "grill", "tread", "stabilizer", "arcades", "panel", "cabinet", "coach roof", "wall", "mattress", "rock", "chair", "leaf", "blinds", "sidewalk", "shelves", "basket", "gas pump", "fruit", "microphone", "wall socket", "light", "elevator", "folding door", "grille door", "sand", "ladder", "ceiling", "curtain", "blind", "clock", "net", "flower", "box", "ladder", "arch", "bottle rack", "skylight", "candle", "steering wheel", "lamp", "waterfall", "floor", "bannister", "roof", "microwave", "plane", "paper", "telephone", "curb", "ramp", "finger", "tree", "bag", "partition", "sconce", "drawing", "bucket", "baseboard", "can", "hovel", "pillar", "rope", "ground", "bottle", "vase", "fluorescent"],
                            # "scene":["street-s", "bedroom-s", "living_room-s", "bathroom-s", "kitchen-s", "dining_room-s", "skyscraper-s", "highway-s", "building_facade-s", "conference_room-s", "hotel_room-s", "mountain_snowy-s", "office-s", "corridor-s", "airport_terminal-s", "game_room-s", "waiting_room-s", "home_office-s", "poolroom-home-s", "art_studio-s", "attic-s", "forest-broadleaf-s", "park-s", "mountain-s", "coast-s", "alley-s", "parlor-s", "closet-s", "beach-s", "childs_room-s", "art_gallery-s", "apartment_building-outdoor-s", "castle-s", "staircase-s", "pasture-s", "dorm_room-s", "nursery-s", "garage-indoor-s", "lobby-s", "reception-s", "bar-s", "forest-needleleaf-s", "warehouse-indoor-s", "bakery-shop-s", "roundabout-s", "house-s", "casino-indoor-s", "classroom-s", "field-cultivated-s", "bridge-s", "river-s", "youth_hostel-s", "field-wild-s", "lighthouse-s", "creek-s", "museum-indoor-s", "window_seat-s", "shoe_shop-s", "amusement_park-s", "dinette-vehicle-s", "lake-natural-s", "dinette-home-s", "cockpit-s", "jacuzzi-indoor-s", "playroom-s", "valley-s", "parking_lot-s", "tower-s", "auditorium-s", "beauty_salon-s", "wet_bar-s", "artists_loft-s", "balcony-interior-s", "arrival_gate-outdoor-s", "plaza-s", "playground-s", "hill-s", "clothing_store-s", "pantry-s", "bow_window-indoor-s", "utility_room-s", "galley-s", "basement-s", "bookstore-s", "abbey-s", "golf_course-s", "supermarket-s", "hallway-s", "market-outdoor-s", "laundromat-s", "greenhouse-indoor-s", "subway_interior-s", "gazebo-exterior-s", "ballroom-s", "alcove-s", "doorway-indoor-s", "bus_interior-s", "access_road-s", "archive-s", "landing_deck-s", "waterfall-block-s", "gymnasium-indoor-s", "office_building-s", "ocean-s", "cathedral-indoor-s", "badlands-s", "forest_path-s", "baggage_claim-s", "home_theater-s", "toyshop-s", "harbor-s", "library-indoor-s", "dentists_office-s", "bowling_alley-s", "amusement_arcade-s", "restaurant-s", "fastfood_restaurant-s", "poolroom-establishment-s", "church-indoor-s", "auto_showroom-s", "gas_station-s", "theater-indoor_procenium-s", "construction_site-s", "fairway-s", "parking_garage-indoor-s", "yard-s", "courthouse-s", "bow_window-outdoor-s", "driveway-s", "water_tower-s", "cubicle-office-s", "computer_room-s", "music_studio-s", "amphitheater-s", "campus-s", "kindergarden_classroom-s", "control_tower-outdoor-s", "ball_pit-s", "desert-sand-s", "cloister-indoor-s", "weighbridge-s", "swimming_pool-outdoor-s", "wine_cellar-barrel_storage-s", "delicatessen-s", "shopping_mall-indoor-s", "restaurant_patio-s", "berth-s", "swimming_pool-indoor-s", "waterfall-fan-s", "videostore-s", "cemetery-s", "reading_room-s", "geodesic_dome-outdoor-s", "windmill-s", "ice_skating_rink-indoor-s", "atrium-public-s", "courtroom-s", "dining_car-s", "day_care_center-s", "movie_theater-indoor-s", "mansion-s", "jail_cell-s", "sauna-s", "mosque-outdoor-s", "lecture_room-s", "campsite-s", "doorway-outdoor-s", "church-outdoor-s", "inn-indoor-s", "hospital_room-s", "florist_shop-indoor-s", "aqueduct-s", "forest_road-s", "cabin-outdoor-s", "operating_room-s", "inn-outdoor-s", "crosswalk-s", "sandbox-s", "hayfield-s", "snowfield-s", "planetarium-outdoor-s", "slum-s", "islet-s", "television_studio-s", "warehouse-outdoor-s", "balcony-exterior-s", "arrival_gate-indoor-s", "plaza-outdoor-s", "cloister-outdoor-s", "weighbridge-outdoor-s", "swimming_pool-indoor_public-s", "wine_cellar-bottle_storage-s", "delicatessen-outdoor-s", "restaurant_patio-outdoor-s", "berth-outdoor-s", "swimming_pool-indoor_public-s", "waterfall-fan-s", "videostore-outdoor-s", "cemetery-outdoor-s", "reading_room-outdoor-s", "geodesic_dome-indoor-s", "windmill-outdoor-s", "ice_skating_rink-outdoor-s", "atrium-public-outdoor-s", "courtroom-outdoor-s", "dining_car-outdoor-s", "day_care_center-outdoor-s", "movie_theater-outdoor-s", "mansion-outdoor-s", "jail_cell-outdoor-s", "sauna-outdoor-s", "mosque-indoor-s", "lecture_room-outdoor-s", "campsite-outdoor-s", "doorway-indoor-s", "church-indoor-s", "inn-outdoor-outdoor-s", "hospital_room-outdoor-s"],
                            # "object":["wall", "sky", "floor", "tree", "building", "person", "ceiling", "table", "windowpane", "road", "grass", "chair", "car", "plant", "painting", "door", "sidewalk", "light", "cabinet", "signboard", "lamp", "ground", "curtain", "pole", "mountain", "fence", "streetlight", "bed", "sofa", "box", "earth", "bottle", "water", "cushion", "book", "flower", "shelf", "carpet", "mirror", "vase", "flowerpot", "sink", "dog", "armchair", "rock", "wall socket", "sconce", "pillow", "cat", "stairs", "pot", "plate", "railing", "fabric", "clock", "bag", "pillar", "bicycle", "coffee table", "ashcan", "bench", "spotlight", "boat", "basket", "work surface", "desk", "bowl", "bird", "house", "plaything", "sea", "paper", "television", "chandelier", "pottedplant", "path", "cup", "stairway", "switch", "van", "stove", "truck", "airplane", "awning", "chest of drawers", "traffic light", "seat", "poster", "flag", "drinking glass", "telephone", "towel", "rope", "tvmonitor", "bush", "bucket", "field", "stool", "tray", "bannister", "fireplace", "trade name", "wood", "fan", "horse", "motorbike", "train", "pedestal", "shelves", "bathtub", "toilet", "refrigerator", "counter", "wardrobe", "food", "candlestick", "computer", "double door", "palm", "bus", "board", "blind", "microwave", "sand", "track", "fluorescent", "sculpture", "countertop", "swivel chair", "snow", "river", "air conditioner", "ottoman", "jar", "step", "pack", "bridge", "minibike", "loudspeaker", "figurine", "skyscraper", "can", "sheep", "bookcase", "bedclothes", "exhaust hood", "hill", "cow", "napkin", "doorframe", "manhole", "umbrella", "case", "autobus", "dishwasher", "container", "text", "knife", "crt screen", "pipe", "magazine", "curb", "oven", "monitoring device", "radiator", "platform", "blanket", "central reservation", "bulletin board", "laptop", "gate", "shoe", "pool table", "metal", "fruit", "keyboard", "coffee maker", "faucet", "candle", "handbag", "wineglass", "mug", "apparel", "land", "pitcher", "remote control", "bar", "ball", "grill", "grandstand", "backpack", "pane of glass", "guitar", "system", "place mat", "bouquet", "rocking chair", "baseboard", "elevator", "ice", "playground", "guardrail", "table tennis", "roof", "bidet", "sweater", "aircraft carrier", "valley", "deck", "railway", "balloon", "pitch", "display board", "mattress", "pallet", "patio", "ad", "price tag", "controls", "island", "stretcher", "curtains", "duck", "pulpit", "dome", "slope", "podium", "ramp", "gas pump", "monument", "trailer", "windmill", "leaves", "helicopter", "pool", "water tower", "folding screen", "workbench", "brushes", "finger", "scoreboard", "ice rink", "carport", "gravestone", "straw", "horse-drawn carriage", "tunnel", "cannon", "tumble dryer", "altarpiece", "shelter", "pond", "windscreen", "leaf", "wheelchair", "coat", "planter", "player", "carousel", "display window", "elevator door", "shop", "roundabout", "bottle rack", "skirt", "aquarium", "ruins", "instrument panel", "ring", "table game", "television camera", "goal", "bird cage", "aqueduct", "weighbridge", "control tower", "blinds", "balcony", "steering wheel", "glass", "stands", "porch", "mill", "pantry", "folding door", "bandstand", "videos", "sand trap", "organ", "synthesizer", "planks", "pictures", "parterre", "lockers", "service station", "trench", "barrels", "box office", "binder", "cabin", "base", "forklift", "pavilion", "brick", "tile", "greenhouse", "caravan", "berth", "trellis", "tomb", "structure", "plastic", "parasol", "dam", "tracks", "hay", "hen", "recycling bin", "disc case", "slide", "shanties", "machinery", "dashboard", "dental chair", "parking", "sewing machine", "rifle", "desert", "henhouse", "tennis court", "shed", "bird feeder", "washing machines", "watchtower", "shops", "ride", "telescope", "drum", "oar", "breads", "semidesert ground", "roller coaster", "water wheel", "barbecue", "bulldozer", "steam shovel", "gravel", "meter", "excavator", "vineyard", "rubble", "badlands", "forest", "ticket counter", "grille door", "stalls", "shower curtain", "village", "gas station", "niche", "check-in-desk", "set of instruments", "bread rolls", "tap", "inflatable bounce game", "temple", "bowling alley", "mosque", "skittle alley", "sandbox", "catwalk", "big top", "iceberg", "viaduct", "fog bank", "parking lot", "trestle", "table cloth", "tables", "pigeonhole", "cactus", "bathrobe", "crate", "quay", "hand cart", "candies", "labyrinth", "bullring", "acropolis", "covered bridge", "shipyard", "elephant", "toll booth", "book stand", "skeleton", "baptismal font", "witness stand", "vegetables", "mountain pass", "meat", "rudder", "terraces", "fire", "panel", "water tank", "headboard", "ticket window", "engine", "sill", "lid", "revolving door", "ear", "screen", "apron", "cockpit", "metal shutters", "cardboard", "casing", "side", "knob", "grid", "handle", "top", "rim", "leather", "fur", "seat cushion", "leg", "seat base", "hair", "bumper", "spindle", "body"]
                            # 
                            }


broden_text_dataset_templates = ["{concept}", "A {concept}", "The {concept}", "A picture of {concept}", "A photo of {concept}"]

class BrodenConceptDataset(torch.utils.data.Dataset):
    def __init__(self,category, target):
        self.category = category
        self.target_concept = target

        self.negative_concepts = [concept for concept in broden_concept_spaces[category] if concept != target]

    def format_concept(self,concept):
        if "-" in concept:
            return concept.split("-")[0]
        else:
            return concept

    def __len__(self):
        return len(broden_text_dataset_templates)

    def __getitem__(self, idx):
        template = broden_text_dataset_templates[idx]
        return template.format(concept=self.format_concept(self.target_concept)), None

class BrodenDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='datasets/broden', resolution=384,
            split='train', category=None,target=None,
            transform=None, transforms=None,
            download=False, size=None,
            broden_version=1, max_segment_depth=6,n_samples=16,seed=None):
        

        #assert category == None or negative_category == None , "Category or Negative category must be None"
        assert resolution in [224, 227, 384]

        ### Our use case only includes a single category.... or negative category: everything except the specified
        categories = [category]
        negative_categories = None #[negative_category]

        if seed != None:
            random.seed(seed)
        
        if download:
            ensure_broden_downloaded(root_dir, resolution, broden_version)
        self.directory = root_dir
        self.resolution = resolution
        self.resdir = os.path.join(root_dir, 'broden%d_%d' %
                (broden_version, resolution))
        self.loader = default_loader
        
        self.transform = transform
        self.transforms = transforms
        
        
        #self.include_bincount = include_bincount
        # The maximum number of multilabel layers that coexist at an image.
        self.max_segment_depth = max_segment_depth
        with open(os.path.join(self.resdir, 'category.csv'),
                encoding='utf-8') as f:
            self.category_info = OrderedDict()
            for row in csv.DictReader(f):
                self.category_info[row['name']] = row
        
        
        if categories is not None:
            # Filter out unused categories
            categories = set([c for c in categories if c in self.category_info])
            for cat in list(self.category_info.keys()):
                if cat not in categories:
                    del self.category_info[cat]

        elif negative_categories is not None:
            # Filter out specified category... leave the rest
            negative_categories = set([c for c in negative_categories if c in self.category_info])
            for cat in list(self.category_info.keys()):
                if cat in negative_categories:
                    del self.category_info[cat]



        categories = list(self.category_info.keys())
        self.categories = categories

        # Filter out unneeded images.
        with open(os.path.join(self.resdir, 'index.csv'),
                encoding='utf-8') as f:
            all_images = [decode_index_dict(r) for r in csv.DictReader(f)]
        self.image = [row for row in all_images
            if index_has_any_data(row, categories) and row['split'] == split]
                
        if size is not None:
            self.image = self.image[:size]


        with open(os.path.join(self.resdir, 'label.csv'),
                encoding='utf-8') as f:
            self.label_info = build_dense_label_array([
                decode_label_dict(r) for r in csv.DictReader(f)])
            self.labels = [l['name'] for l in self.label_info]

        # Build dense remapping arrays for labels, so that you can
        # get dense ranges of labels for each category.
        self.category_map = {}
        self.category_unmap = {}
        self.category_label = {}
        self.target_code = None
        for cat in self.categories:
            with open(os.path.join(self.resdir, 'c_%s.csv' % cat),
                    encoding='utf-8') as f:
                c_data = [decode_label_dict(r) for r in csv.DictReader(f)]
            self.category_unmap[cat], self.category_map[cat] = (
                    build_numpy_category_map(c_data))
            self.category_label[cat] = build_dense_label_array(
                    c_data, key='code')

            for label in self.category_label[cat]:
                if label["name"] == target:
                    self.target_code = label["number"]


        self.num_labels = len(self.labels)
        # Primary categories for each label is the category in which it
        # appears with the maximum coverage.
        self.label_category = numpy.zeros(self.num_labels, dtype=int)
        for i in range(self.num_labels):
            maxcoverage, self.label_category[i] = max(
               (self.category_label[cat][self.category_map[cat][i]]['coverage']
                    if i < len(self.category_map[cat])
                       and self.category_map[cat][i] else 0, ic)
                for ic, cat in enumerate(categories))

        random.shuffle(self.image)

        _image = []

        for record in self.image:
            mask = self.load_mask(record)

            if mask.mean() > 0.1:
                _image.append(record)

            if len(_image) >= n_samples:
                break


        self.image = _image

    def __len__(self):
        return len(self.image)

    def load_mask(self,record):
        mask = numpy.zeros(shape=(
            record['sh'], record['sw']), dtype=int)
        
        mask = torch.zeros((record['sh'], record['sw']))

        depth = 0
        for cat in self.categories:
            for layer in record[cat]:
                    if isinstance(layer, int):             
                        if layer == self.target_code:
                            mask = mask + 1
                    else:
                        png = torch.Tensor(numpy.asarray(self.loader(os.path.join(
                            self.resdir, 'images', layer))))

                        #print(png[:,:,0].amax())
                        #print(png[:,:,0].amin())

                        _mask = (png[:,:,0] + png[:,:,1]).squeeze(-1)

                        #print(f"_mask: {_mask.shape} {self.target_code}")

                        mask[:,:] = (_mask == self.target_code).to(torch.int)# ((_mask - _mask.amin())/(_mask.amax() - _mask.amin())).to(torch.float)

                    depth += 1
        
        mask = torch.Tensor(mask)
        mask = transforms.Resize(self.resolution)(mask.unsqueeze(0)).squeeze(0)

        return mask
    
    def __getitem__(self, idx):
        record = self.image[idx]
        # example record: {
        #    'image': 'opensurfaces/25605.jpg', 'split': 'train',
        #    'ih': 384, 'iw': 384, 'sh': 192, 'sw': 192,
        #    'color': ['opensurfaces/25605_color.png'],
        #    'object': [], 'part': [],
        #    'material': ['opensurfaces/25605_material.png'],
        #    'scene': [], 'texture': []}


        image = self.loader(os.path.join(self.resdir, 'images',
            record['image']))

        image = transforms.ToTensor()(image)

        mask = self.load_mask(record)

        if self.transforms != None:            
            comb = self.transforms(torch.cat([image,mask.unsqueeze(0)]))
            image = comb[:3]
            mask = comb[3]

        if self.transform != None:
            image = self.transform(image)


        return (image, mask.unsqueeze(0))

    def plot(self,n_samples = 4):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(n_samples, 2, squeeze=False,figsize=(25, 25))

        for idx,(image,label) in enumerate(self):
            if idx >= n_samples:
                break
            #print(image.shape)
            #print(label.shape)
            ax[idx,0].imshow(image.permute(1,2,0))
            im = ax[idx,1].imshow(label.permute(1,2,0),vmin=0,vmax=1)

            plt.colorbar(im, ax=ax[idx,1])

        plt.show()

    

def build_dense_label_array(label_data, key='number', allow_none=False):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    result = [None] * (max([d[key] for d in label_data]) + 1)
    for d in label_data:
        result[d[key]] = d
    # Fill in none
    if not allow_none:
        example = label_data[0]
        def make_empty(k):
            return dict((c, k if c is key else type(v)())
                    for c, v in example.items())
        for i, d in enumerate(result):
            if d is None:
                result[i] = dict(make_empty(i))
    return result

def build_numpy_category_map(map_data, key1='code', key2='number'):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    results = list(numpy.zeros((max([d[key] for d in map_data]) + 1),
            dtype=numpy.int16) for key in (key1, key2))
    for d in map_data:
        results[0][d[key1]] = d[key2]
        results[1][d[key2]] = d[key1]
    return results

def index_has_any_data(row, categories):
    for c in categories:
        for data in row[c]:
            if data: return True
    return False

def decode_label_dict(row):
    result = {}
    for key, val in row.items():
        if key == 'category':
            result[key] = dict((c, int(n))
                for c, n in [re.match('^([^(]*)\(([^)]*)\)$', f).groups()
                    for f in val.split(';')])
        elif key == 'name':
            result[key] = val
        elif key == 'syns':
            result[key] = val.split(';')
        elif re.match('^\d+$', val):
            result[key] = int(val)
        elif re.match('^\d+\.\d*$', val):
            result[key] = float(val)
        else:
            result[key] = val
    return result

def decode_index_dict(row):
    result = {}
    for key, val in row.items():
        if key in ['image', 'split']:
            result[key] = val
        elif key in ['sw', 'sh', 'iw', 'ih']:
            result[key] = int(val)
        else:
            item = [s for s in val.split(';') if s]
            for i, v in enumerate(item):
                if re.match('^\d+$', v):
                    item[i] = int(v)
            result[key] = item
    return result

def ensure_broden_downloaded(directory, resolution, broden_version=1):
    assert resolution in [224, 227, 384]
    baseurl = 'http://netdissect.csail.mit.edu/data/'
    dirname = 'broden%d_%d' % (broden_version, resolution)
    if os.path.isfile(os.path.join(directory, dirname, 'index.csv')):
        return # Already downloaded
    zipfilename = 'broden1_%d.zip' % resolution
    download_dir = os.path.join(directory, 'download')
    os.makedirs(download_dir, exist_ok=True)
    full_zipfilename = os.path.join(download_dir, zipfilename)
    if not os.path.exists(full_zipfilename):
        url = '%s/%s' % (baseurl, zipfilename)
        print('Downloading %s' % url)
        data = urlopen(url)
        with open(full_zipfilename, 'wb') as f:
            f.write(data.read())
    print('Unzipping %s' % zipfilename)
    with zipfile.ZipFile(full_zipfilename, 'r') as zip_ref:
        zip_ref.extractall(directory)
    assert os.path.isfile(os.path.join(directory, dirname, 'index.csv'))

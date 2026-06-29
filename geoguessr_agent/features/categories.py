"""
Feature categories aligned with the Plonkit knowledge base.
Each category maps to CLIP text prompts for zero-shot feature extraction.
"""

from typing import NamedTuple


class ClueCategory(NamedTuple):
    key: str
    label: str
    prompts: list[tuple[str, str]]


CATEGORY_PROMPTS: list[ClueCategory] = [
    ClueCategory(
        key="driving_side",
        label="Driving side",
        prompts=[
            ("left", "cars driving on the left side of the road"),
            ("right", "cars driving on the right side of the road"),
        ],
    ),
    ClueCategory(
        key="road_lines",
        label="Road lines",
        prompts=[
            ("white_outer_yellow_middle", "road with white outer lines and yellow middle lines"),
            ("double_yellow_middle", "road with double solid yellow middle lines"),
            ("all_white", "road with only white road lines, no yellow lines"),
            ("triple_white", "road with three white centre lines, two solid and one dashed"),
            ("yellow_outer_white_middle", "road with yellow outer lines and white centre lines"),
            ("dashed_yellow_middle", "road with a single dashed yellow middle line"),
            ("solid_white_outer_dashed_middle",
             "road with solid white outer lines and white dashed middle"),
        ],
    ),
    ClueCategory(
        key="bollards",
        label="Bollards",
        prompts=[
            ("black_white_striped", "a black and white striped roadside bollard, pointed top"),
            ("yellow_bollard", "a yellow roadside bollard with a white reflector"),
            ("white_red_reflector", "a white bollard with a red reflector on the front"),
            ("white_grey_reflector", "a white bollard with a grey reflector"),
            ("wedge_shaped", "a wedge-shaped flat roadside bollard"),
            ("round_concrete_white", "a round white concrete bollard with circular reflectors"),
            ("orange_reflector", "a bollard with an orange reflector near an intersection"),
            ("no_bollards", "a road with no visible roadside bollards or delineator posts"),
        ],
    ),
    ClueCategory(
        key="poles",
        label="Utility poles",
        prompts=[
            ("round_concrete", "round concrete utility pole with horizontal crossbars"),
            ("square_concrete", "square concrete utility pole with vertical holes or engravings"),
            ("wooden_small_cap", "wooden utility pole with a small black metal cap on top"),
            ("wooden_only", "plain wooden utility pole without crossbars or insulators"),
            ("trident_top", "utility pole with a trident shaped pole top, three insulating pins"),
            ("ladder_pole", "rectangular concrete pole with step-like ladder indentations"),
            ("stobie_pole", "stobie pole: steel I-beam and concrete, rectangular shape"),
            ("L_shaped_crossbar", "utility pole with an L-shaped thin metal crossbar"),
            ("triangle_top", "utility pole with triangle shaped pole top, inset insulators"),
            ("bird_poles",
             "utility pole with many horizontal bars, thin white bird-like insulators"),
            ("no_poles", "an area with no visible utility poles or overhead power lines"),
        ],
    ),
    ClueCategory(
        key="guardrails",
        label="Guardrails",
        prompts=[
            ("white_guardrail", "a white metal guardrail along the road"),
            ("yellow_reflector_guardrail", "a guardrail with yellow rectangular reflectors"),
            ("red_yellow_ending", "a guardrail ending with red and yellow striped stickers"),
            ("B_type_guardrail", "B-type guardrail: rounded metal posts, undulating face"),
            ("A_type_guardrail", "A-type guardrail: flatter metal face, different post shape"),
            ("no_guardrail", "a road without any visible guardrails or safety barriers"),
        ],
    ),
    ClueCategory(
        key="signage_script",
        label="Signage script",
        prompts=[
            ("latin", "traffic signs and billboards with Latin alphabet text"),
            ("kanji_hiragana", "signs with Japanese kanji and hiragana characters"),
            ("hangul", "signs with Korean hangul characters"),
            ("thai", "signs with Thai script characters"),
            ("cyrillic", "signs with Cyrillic alphabet text"),
            ("arabic", "signs with Arabic script text"),
            ("devanagari", "signs with Devanagari script text"),
            ("chinese", "signs with Chinese characters only, no hiragana"),
        ],
    ),
    ClueCategory(
        key="architecture",
        label="Architecture",
        prompts=[
            ("red_tiled_roofs", "buildings with orange or red clay tiled roofs"),
            ("dark_red_wooden", "dark red painted wooden houses with white trim"),
            ("white_concrete_flat", "white concrete apartment buildings with flat roofs"),
            ("corrugated_metal_roof", "buildings with gray corrugated metal sheet roofs"),
            ("brick_houses", "red brick houses with steep roofs"),
            ("half_timbered", "half-timbered houses, dark wooden beams and white plaster"),
            ("modern_glass_steel", "modern glass and steel highrise buildings"),
            ("no_buildings", "rural or remote area with no visible buildings"),
        ],
    ),
    ClueCategory(
        key="landscape",
        label="Landscape",
        prompts=[
            ("flat_treeless", "flat open terrain with few or no trees, wide empty landscape"),
            ("rolling_hills_grass", "rolling green hills with grass and scattered trees"),
            ("mountains_rocky", "rocky mountains with exposed rock faces"),
            ("boreal_forest", "dense boreal forest with pine and spruce trees"),
            ("tropical_vegetation", "tropical dense vegetation with palm trees and broad leaves"),
            ("red_soil", "landscape with distinctive red or reddish-orange soil"),
            ("dark_volcanic",
             "barren landscape, dark volcanic rock and gravel, minimal vegetation"),
            ("desert_arid", "arid desert landscape with sand, cacti, or dry shrubs"),
        ],
    ),
    ClueCategory(
        key="coverage",
        label="Coverage gen",
        prompts=[
            ("low_cam", "street view from a low-mounted camera with large frontal blur"),
            ("small_cam", "street view with small distinct circular camera blur, standard gen 4"),
            ("gen2_cam", "street view with large circular blur creating a halo effect"),
            ("no_blur_visible", "street view with no visible camera blur or car meta"),
        ],
    ),
    ClueCategory(
        key="vegetation",
        label="Vegetation",
        prompts=[
            ("eucalyptus", "eucalyptus trees, light colored smooth bark, puffy leaf clusters"),
            ("palm_trees", "palm trees with fan-shaped or feather-like fronds"),
            ("pine_trees", "pine trees with tall straight trunks and dark green needles"),
            ("broadleaf_deciduous", "broadleaf deciduous trees with dense rounded canopy"),
            ("bamboo", "bamboo plants with tall thin stalks and narrow leaves"),
            ("agricultural_crops", "farmland with agricultural crops in rows"),
        ],
    ),
    ClueCategory(
        key="license_plates",
        label="License plates",
        prompts=[
            ("long_white_eu_strip",
             "long white license plates with a blue European strip on the left"),
            ("no_eu_strip", "long white license plates without any blue European strip"),
            ("short_white_green_text", "short white license plates with green text"),
            ("white_blue_stripe_top", "white license plates with a blue stripe at the top"),
            ("yellow_rear", "yellow colored license plates on vehicles"),
            ("not_visible", "vehicles with no clearly visible license plates"),
        ],
    ),
]

CATEGORY_NAMES = [cat.key for cat in CATEGORY_PROMPTS]


def get_feature_dim() -> int:
    return sum(len(cat.prompts) for cat in CATEGORY_PROMPTS)

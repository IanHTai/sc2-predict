NA = ["United States", "Canada", "Mexico"]
SA = ["Argentina",
"Belize",
"Bolivia",
"Brazil",
"Chile",
"Colombia",
"Costa Rica",
"Ecuador",
"El Salvador",
"Guatemala",
"Honduras",
"Jamaica",
"Mexico",
"Nicaragua",
"Panama",
"Paraguay",
"Peru",
"Puerto Rico",
"Uruguay",
"Haiti",
"Dominican Republic",
"Venezuela"]



EUW = [
    "Algeria",
    "Austria",
    "Bahrain",
    "Belarus",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Egypt",
    "Estonia",
    "Finland",
    "France",
    "Georgia",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Israel",
    "Italy",
    "Kazakhstan",
    "Latvia",
    "Lebanon",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Morocco",
    "Netherlands",
    "Norway",
    "Poland",
    "Portugal",
    "Qatar",
    "Moldova",
    "Romania",
    "Russia",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "Spain",
    "Sweden",
    "Switzerland",
    "Tunisia",
    "Turkey",
    "Ukraine",
    "United Kingdom"
]

KR = ["Korea Republic of"]
ASIA = ["Australia", "New Zealand", "Indonesia", "Malaysia", "Singapore", "Thailand", "Philippines", "Vietnam",
        "Taiwan", "Hong Kong", "Macau", "Japan", "China"]


REGIONS = {
    'NA': NA,
    'SA': SA,
    'EUW': EUW,
    'KR': KR,
}

REGION_DICT = {}

def fillRegionDict():
    for country in NA + SA:
        REGION_DICT[country] = 'NA'
    # for country in SA:
    #     REGION_DICT[country] = 'SA'
    for country in EUW:
        REGION_DICT[country] = 'EUW'
    # for country in KR:
    #     REGION_DICT[country] = 'KR'
    for country in ASIA + KR:
        REGION_DICT[country] = 'ASIA'
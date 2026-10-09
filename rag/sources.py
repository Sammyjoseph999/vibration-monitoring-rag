"""The pages the assistant reads, and the questions used to check retrieval."""

SOURCES = [
    # Vendor pages for vibration monitoring products
    {"name": "Fluke", "url": "https://www.fluke.com/en-us/products/condition-monitoring/vibration"},
    {"name": "Erbessd Instruments", "url": "https://www.erbessd-instruments.com/"},
    {"name": "Acoem", "url": "https://www.acoem.com/en/products/condition-monitoring/portable-vibration-monitoring/machine-defender-app/"},
    {"name": "Emerson", "url": "https://www.emerson.com/en-us/catalog/ams-a2140"},
    # Background reading on the subject
    {"name": "Wikipedia: Condition monitoring", "url": "https://en.wikipedia.org/wiki/Condition_monitoring"},
    {"name": "Wikipedia: Predictive maintenance", "url": "https://en.wikipedia.org/wiki/Predictive_maintenance"},
    {"name": "Wikipedia: Vibration", "url": "https://en.wikipedia.org/wiki/Vibration"},
    {"name": "Wikipedia: Accelerometer", "url": "https://en.wikipedia.org/wiki/Accelerometer"},
]

# Each question names the source that should appear among the retrieved chunks.
EVAL_QUESTIONS = [
    ("What vibration monitoring tools does Fluke offer?", "Fluke"),
    ("What is the Machine Defender app used for?", "Acoem"),
    ("What is the AMS 2140 machinery health analyzer?", "Emerson"),
    ("What products does Erbessd Instruments make?", "Erbessd Instruments"),
    ("What is condition monitoring?", "Wikipedia: Condition monitoring"),
    ("How does predictive maintenance differ from preventive maintenance?", "Wikipedia: Predictive maintenance"),
    ("What is the difference between free and forced vibration?", "Wikipedia: Vibration"),
    ("How does a piezoelectric accelerometer work?", "Wikipedia: Accelerometer"),
    ("What is resonance?", "Wikipedia: Vibration"),
    ("Which techniques are used to monitor rotating machinery?", "Wikipedia: Condition monitoring"),
]

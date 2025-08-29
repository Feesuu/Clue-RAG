EXTRACTION_PROMPT = """
Decompose the "Content" into clear and simple knowledge units, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Pronoun Elimination: Replace ALL pronouns (it, they, this, etc.) with full taxonomic names or explicit references. Never use possessives (its, their), always use "[entity]'s [property]" construction.
4. Present the results as a list of strings, formatted in JSON.

Example-1:
Input: Jes\u00fas Aranguren. His 13-year professional career was solely associated with Athletic Bilbao, with which he played in nearly 400 official games, winning two Copa del Rey trophies.
Output: {{ "knowledge_units": [ "Jesús Aranguren had a 13-year professional career.", "Jesús Aranguren's professional career was solely associated with Athletic Bilbao.", "Athletic Bilbao is a football club.", "Jesús Aranguren played for Athletic Bilbao in nearly 400 official games.", "Jesús Aranguren won two Copa del Rey trophies with Athletic Bilbao."]}}

Example-2:
Input: Ophrys apifera. Ophrys apifera grows to a height of 15 -- 50 centimetres (6 -- 20 in). This hardy orchid develops small rosettes of leaves in autumn. They continue to grow slowly during winter. Basal leaves are ovate or oblong - lanceolate, upper leaves and bracts are ovate - lanceolate and sheathing. The plant blooms from mid-April to July producing a spike composed from one to twelve flowers. The flowers have large sepals, with a central green rib and their colour varies from white to pink, while petals are short, pubescent, yellow to greenish. The labellum is trilobed, with two pronounced humps on the hairy lateral lobes, the median lobe is hairy and similar to the abdomen of a bee. It is quite variable in the pattern of coloration, but usually brownish - red with yellow markings. The gynostegium is at right angles, with an elongated apex.
Output: {{ "knowledge_units": [ "Ophrys apifera grows to a height of 15-50 centimetres (6-20 in)", "Ophrys apifera is a hardy orchid", "Ophrys apifera develops small rosettes of leaves in autumn", "The leaves of Ophrys apifera continue to grow slowly during winter", "The basal leaves of Ophrys apifera are ovate or oblong-lanceolate", "The upper leaves and bracts of Ophrys apifera are ovate-lanceolate and sheathing", "Ophrys apifera blooms from mid-April to July", "Ophrys apifera produces a spike composed of one to twelve flowers", "The flowers of Ophrys apifera have large sepals with a central green rib", "The flowers of Ophrys apifera vary in colour from white to pink", "The petals of Ophrys apifera are short, pubescent, and yellow to greenish", "The labellum of Ophrys apifera is trilobed with two pronounced humps on the hairy lateral lobes", "The median lobe of Ophrys apifera's labellum is hairy and resembles a bee's abdomen", "The coloration pattern of Ophrys apifera is variable but usually brownish-red with yellow markings", "The gynostegium of Ophrys apifera is at right angles with an elongated apex" ]}}

JUST OUTPUT THE RESULTS IN JSON FORMAT! DON'T OUTPUT ANYTHING INRELEVENT!
Input: {passage}
Output:
"""

NER_PROMPT = """
Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
Example:
Input: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
Output: {{"named_entities":["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}}

Input: {passage}
Output:
"""

GENERATION_PROMPT = """
Your goal is to give the best full answer to question the user input according to the given context below.
Given Context: {context_data}

Give the best full answer to question: {question}.

Answer this question in as fewer number of words as possible. Don't output your thinking process!
JUST OUTPUT THE ANSWER BELOW WITH THE PREFIX "ANSWER", SUCH AS "ANSWER: <YOUR OUTPUT>".
"""

# SYSTEM_PROMPT = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
 
# {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
import os
import warnings
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pinecone import Pinecone
import google.generativeai as genai
from flask import Flask, request, jsonify
import logging


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

FLASK_DEBUG = 1
PINECONE_API_KEY="1d8e304a-f268-4aa2-b6e3-21c2c634bfee"
GOOGLE_AI_API_KEY="AIzaSyBBco-KQ3V4QsocFHhPKHF_5kchVRzIoFU"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Pinecone
api_key = PINECONE_API_KEY
pc = Pinecone(api_key=api_key)
index_name = "hybrid-search-langchain-pinecone-5"
index = pc.Index(index_name)

# Initialize embeddings 
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", clean_up_tokenization_spaces=True)

# Initialize Google AI
genai.configure(api_key=GOOGLE_AI_API_KEY)

# Custom synonyms dictionary
custom_synonyms = {
   "money": ["cash", "currency", "funds", "finance", "capital"],
"cross-border": ["international", "transnational", "global", "overseas", "foreign"],
"payment": ["transaction", "remittance", "transfer", "exchange", "settlement"],
"platform": ["system", "service", "solution", "portal", "interface"],
"business": ["company", "enterprise", "organization", "firm", "corporation"],
"offer": ["provide", "deliver", "give", "present", "supply"],
"service": ["feature", "function", "facility", "utility", "amenity"],
"currency": ["money", "cash", "funds", "exchange", "finance"],
"conversion": ["exchange", "translation", "transformation", "transference", "substitution"],
"collection": ["gathering", "accumulation", "assemblage", "aggregation", "compilation"],
"remittance": ["transfer", "transmission", "dispatch", "conveyance", "shipment"],
"invoicing": ["billing", "charging", "pricing", "quoting", "estimation"],
"solution": ["answer", "resolution", "fix", "remedy", "strategy"],
"available": ["accessible", "obtainable", "provided", "on-hand", "ready"],
"launching": ["introducing", "unveiling", "debuting", "premiering", "initiating"],
"freelancer": ["independent contractor", "self-employed", "solopreneur", "entrepreneur", "consultant"],
"restriction": ["limitation", "constraint", "control", "regulation", "requirement"],
"regulation": ["rule", "policy", "law", "guideline", "protocol"],
"receive": ["obtain", "acquire", "get", "gain", "accept"],
"foreign": ["international", "overseas", "global", "external", "cross-border"],
"payment": ["transaction", "remittance", "transfer", "exchange", "settlement"],
"convert": ["exchange", "transform", "translate", "swap", "change"],
"indian": ["domestic", "national", "local", "in-country", "inland"],
"bank": ["financial institution", "lender", "depository", "credit union", "savings"],
"account": ["ledger", "record", "statement", "profile", "portfolio"],
"secure": ["safe", "protected", "guarded", "shielded", "assured"],
"regulate": ["control", "govern", "oversee", "supervise", "monitor"],
"partner": ["ally", "collaborator", "associate", "affiliate", "joint venture"],
"fee": ["charge", "cost", "price", "rate", "tariff"],
"withdraw": ["retrieve", "remove", "extract", "take out", "pull out"],
"document": ["paperwork", "record", "file", "credential", "proof"],
"beneficial": ["advantageous", "helpful", "useful", "valuable", "favorable"],
"owner": ["proprietor", "holder", "shareholder", "stakeholder", "possessor"],
"access": ["obtain", "retrieve", "gain", "reach", "acquire"],
"immediate": ["instant", "prompt", "quick", "rapid", "swift"],
"credit": ["deposit", "transfer", "post", "add", "put"],
"real-time": ["live", "current", "dynamic", "ongoing", "simultaneous"],
"rate": ["price", "fee", "charge", "cost", "tariff"],
"trusted": ["reliable", "dependable", "credible", "reputable", "trustworthy"],
"provider": ["supplier", "vendor", "source", "distributor", "seller"],
"support": ["assistance", "help", "aid", "guidance", "backing"],
"team": ["staff", "personnel", "workforce", "employees", "crew"],
"guidance": ["advice", "counsel", "direction", "instruction", "recommendation"],
"navigate": ["maneuver", "traverse", "explore", "find one's way", "journey"],
"complex": ["intricate", "complicated", "involved", "elaborate", "convoluted"],
"requirement": ["need", "necessity", "prerequisite", "condition", "demand"],
"taxation": ["taxes", "levies", "duties", "imposts", "fiscal obligations"],
"reporting": ["documenting", "recording", "logging", "tracking", "accounting"],
"mitigate": ["reduce", "minimize", "lessen", "alleviate", "diminish"],
"fluctuation": ["variation", "change", "alteration", "shift", "volatility"],
"encryption": ["coding", "ciphering", "scrambling", "masking", "concealment"],
"transmission": ["sending", "conveying", "transferring", "delivering", "dispatching"],
"balance": ["amount", "quantity", "sum", "value", "total"],
"assistance": ["help", "support", "guidance", "aid", "facilitation"],
"concern": ["issue", "problem", "matter", "question", "difficulty"],
"value-added": ["enhanced", "improved", "premium", "upgraded", "supplementary"],
"trade": ["commerce", "business", "transactions", "buying and selling", "exchange"],
"finance": ["economy", "monetary affairs", "banking", "accounting", "investments"],
"strategy": ["plan", "approach", "method", "tactic", "program"],
"marketplace": ["market", "exchange", "platform", "network", "hub"],
"credit-card": ["charge card", "payment card", "plastic money", "revolving credit", "bank card"],
"digital-asset": ["cryptocurrency", "virtual currency", "blockchain asset", "token", "coin"],
"download": ["retrieve", "obtain", "access", "acquire", "extract"],
"transaction": ["activity", "operation", "exchange", "deal", "transfer"],
"draft": ["proposal", "outline", "sketch", "preliminary version", "incomplete work"],
"edit": ["modify", "revise", "update", "amend", "change"],
"delete": ["remove", "erase", "discard", "eliminate", "get rid of"],
"view": ["see", "observe", "examine", "inspect", "look at"],
"time zone": ["timezone", "time region", "time offset", "time location", "time area"],
"change": ["modify", "alter", "adjust", "transform", "switch"],
"settings": ["preferences", "options", "configurations", "controls", "parameters"],
"payment": ["transaction", "remittance", "transfer", "exchange", "settlement"],
"status": ["state", "condition", "standing", "position", "situation"],
"notify": ["inform", "alert", "advise", "communicate", "signal"],
"invoice": ["bill", "statement", "receipt", "tab", "charge"],
"follow-up": ["follow through", "continue", "pursue", "check on", "revisit"],
"recurring": ["repeating", "periodic", "cyclical", "regular", "recurrent"],
"password": ["passcode", "secret word", "access code", "authentication key", "credential"],
"update": ["modify", "change", "revise", "upgrade", "refresh"],
"currency": ["money", "cash", "funds", "finance", "exchange"],
"exchange": ["conversion", "swap", "trade", "rate", "transform"],
"charge": ["fee", "price", "cost", "rate", "tariff"],
"virtual": ["digital", "simulated", "computer-generated", "online", "electronic"],
"account": ["ledger", "record", "profile", "portfolio", "statement"],
"support": ["assistance", "help", "aid", "guidance", "facilitation"],
"equity": ["shares", "ownership", "stock", "investment", "capital"],
"debt": ["borrowing", "loan", "securities", "obligation", "liability"],
"investment": ["funding", "finance", "capital", "assets", "backing"],
"real estate": ["property", "land", "buildings", "immovable assets", "realty"],
"portfolio": ["collection", "assortment", "set", "group", "array"],
"loan": ["debt", "borrowing", "credit", "finance", "advance"],
"deposit": ["savings", "balance", "funds", "money", "account"],
"overdraft": ["line of credit", "negative balance", "shortfall", "deficit", "advance"],
"intangible": ["intellectual", "non-physical", "virtual", "abstract", "incorporeal"],
"asset": ["resource", "property", "holding", "possession", "investment"],
"shipping": ["transportation", "freight", "cargo", "delivery", "logistics"],
"collection": ["gathering", "accumulation", "assemblage", "aggregation", "compilation"],
"negotiation": ["purchase", "acquisition", "procurement", "trade", "deal"],
"discounting": ["financing", "advance", "prepayment", "factoring", "monetization"],
"remittance": ["transfer", "transmission", "dispatch", "conveyance", "shipment"],
"leasing": ["renting", "hiring", "letting", "chartering", "leasing"],
"freight": ["cargo", "goods", "shipment", "load", "merchandise"],
"passenger": ["traveler", "rider", "commuter", "customer", "voyager"],
"fare": ["ticket price", "cost of travel", "fee", "charge", "tariff"],
"operating": ["functional", "working", "active", "running", "in use"],
"travel": ["trip", "journey", "tour", "vacation", "excursion"],
"communication": ["messaging", "signaling", "transmission", "correspondence", "exchange"],
"postal": ["mail", "delivery", "courier", "dispatch", "shipment"],
"telecommunication": ["telephony", "data transfer", "networking", "broadcasting", "media"],
"satellite": ["space technology", "extraterrestrial", "orbit", "spacecraft", "celestial"],
"construction": ["building", "development", "erection", "engineering", "infrastructure"],
"insurance": ["coverage", "protection", "indemnity", "underwriting", "risk management"],
"premium": ["fee", "charge", "price", "rate", "cost"],
"reinsurance": ["backup insurance", "secondary coverage", "excess insurance", "retrocession", "reassurance"],
"auxiliary": ["supplementary", "additional", "supporting", "complementary", "assisting"],
"claim": ["demand", "request", "application", "entitlement", "assertion"],
"intermediation": ["brokerage", "facilitation", "mediation", "agency", "liaison"],
"investment banking": ["securities", "underwriting", "mergers", "acquisitions", "capital markets"],
"consultancy": ["advisory", "expertise", "guidance", "counsel", "professional services"],
"hardware": ["equipment", "devices", "machinery", "components", "physical infrastructure"],
"software": ["programs", "applications", "code", "digital systems", "technology"],
"database": ["data storage", "information repository", "records", "data organization", "data management"],
"news": ["information", "updates", "intelligence", "reporting", "journalism"],
"franchise": ["license", "distribution rights", "trademark", "branding", "commercial concession"],
"prototype": ["model", "sample", "pilot", "original", "proof-of-concept"],
"merchanting": ["trading", "reselling", "arbitrage", "brokerage", "middleman services"],
"commission": ["fee", "percentage", "charges", "cut", "remuneration"],
"leasing (operational)": ["renting", "hiring", "letting", "chartering", "leasing"],
"legal": ["law", "juridical", "judicial", "statutory", "legislative"],
"accounting": ["auditing", "bookkeeping", "finance", "tax", "reporting"],
"business": ["enterprise", "company", "organization", "firm", "corporation"],
"management": ["administration", "leadership", "oversight", "control", "supervision"],
"public relations": ["communications", "marketing", "branding", "publicity", "outreach"],
"advertising": ["promotion", "marketing", "commercials", "publicity", "outreach"],
"market research": ["consumer analysis", "trend analysis", "data collection", "surveying", "intelligence"],
"polling": ["surveying", "canvassing", "questionnaires", "sampling", "data collection"],
"research & development": ["R&D", "innovation", "experimentation", "investigation", "invention"],
"technical": ["engineering", "scientific", "technological", "specialized", "professional"],
"agricultural": ["farming", "cultivation", "agronomy", "horticulture", "livestock"],
"mining": ["extraction", "quarrying", "prospecting", "digging", "exploration"],
"on-site": ["in-person", "on-location", "at the premises", "at the site", "in-situ"],
"distribution": ["logistics", "supply chain", "delivery", "transportation", "warehousing"],
"environmental": ["ecological", "green", "sustainable", "conservation", "renewable"],
"audio-visual": ["multimedia", "film", "video", "television", "broadcasting"],
"cultural": ["artistic", "literary", "historical", "traditional", "heritage"],
"embassy": ["diplomatic mission", "consulate", "legation", "foreign office", "chancery"],
"institution": ["organization", "establishment", "association", "agency", "body"],
"family": ["relatives", "kin", "household", "dependents", "loved ones"],
"donation": ["gift", "contribution", "offering", "grant", "endowment"],
"refund": ["reimbursement", "rebate", "repayment", "return", "credit"],
"compensation": ["remuneration", "payment", "salary", "wages", "earnings"],
"interest": ["yield", "return", "earnings", "income", "profit"],
"profit": ["earnings", "income", "revenue", "gain", "surplus"],
"dividend": ["share of profits", "distribution", "payout", "yield", "return"],
"rebate": ["discount", "refund", "credit", "reduction", "allowance"],
"remittance": ["transfer", "transmission", "dispatch", "conveyance", "shipment"],
"architectural": ["design", "construction", "engineering", "drafting", "technical"],
"engineering": ["design", "construction", "architectural", "drafting", "technical"],
"technical": ["design", "construction", "engineering", "architectural", "drafting"],
"agricultural": ["farming", "rural", "forestry", "pastoral", "cultivation"],
"mining": ["extraction", "mineral", "ore", "geological", "subsurface"],
"processing": ["production", "manufacturing", "refining", "treatment", "conversion"],
"services": ["assistance", "support", "facilitation", "provision", "operations"],
"disease": ["illness", "infection", "sickness", "malady", "ailment"],
"harvest": ["crop", "yield", "production", "reaping", "gathering"],
"forestry": ["arboriculture", "silviculture", "woodland", "tree farming", "forest management"],
"offices": ["branches", "divisions", "subsidiaries", "facilities", "locations"],
"distribution": ["logistics", "transportation", "delivery", "wholesale", "retail"],
"environmental": ["ecological", "green", "sustainable", "conservation", "protection"],
"other": ["miscellaneous", "assorted", "diverse", "various", "additional"],
"audio-visual": ["media", "entertainment", "production", "distribution", "licensing"],
"personal": ["cultural", "leisure", "recreational", "educational", "social"],
"embassies": ["consulates", "missions", "delegations", "diplomatic posts", "foreign offices"],
"international": ["global", "worldwide", "transnational", "multinational", "intercontinental"],
"maintenance": ["support", "upkeep", "operations", "servicing", "preservation"],
"family": ["household", "relatives", "dependents", "kin", "relations"],
"gifts": ["presents", "donations", "contributions", "offerings", "grants"],
"religious": ["spiritual", "faith-based", "devotional", "ecclesiastical", "sacred"],
"charitable": ["philanthropic", "benevolent", "humanitarian", "altruistic", "welfare"],
"government": ["state", "public", "official", "administrative", "civil"],
"employees": ["staff", "personnel", "workers", "labor force", "workforce"],
"interest": ["dividends", "income", "returns", "earnings", "profits"],
"debt": ["loans", "bonds", "securities", "credit", "financing"],
"imports": ["acquisitions", "purchases", "inbound goods", "incoming shipments", "brought-in items"],
"refunds": ["rebates", "reimbursements", "returns", "repayments", "compensation"],
"international": ["global", "worldwide", "transnational", "multinational", "intercontinental"],
"bidding": ["tendering", "procurement", "contracting", "proposals", "competitive selection"],
"equity": ["shares", "stocks", "ownership", "equities", "securities"],
"branches": ["subsidiaries", "affiliates", "divisions", "offices", "locations"],
"associates": ["partners", "joint ventures", "affiliates", "subsidiaries", "related entities"],
"real estate": ["property", "land", "buildings", "realty", "immovable assets"],
"debt securities": ["bonds", "debentures", "notes", "fixed income instruments", "credit instruments"],
"dividends": ["distributions", "payouts", "earnings", "income", "returns"],
"loans": ["credit", "financing", "debt", "borrowing", "advances"],
"deposits": ["accounts", "savings", "investments", "placements", "funds"],
"exports": ["outbound goods", "shipments", "merchandise", "commodities", "foreign sales"],
"freight": ["cargo", "shipments", "transportation", "haulage", "logistics"],
"passenger": ["travelers", "commuters", "tourists", "riders", "patrons"],
"travel": ["tourism", "trips", "vacations", "journeys", "expeditions"],
"postal": ["mail", "courier", "shipping", "delivery", "logistics"],
"intangible assets": ["intellectual property", "patents", "copyrights", "trademarks", "goodwill"],
"freight": ["cargo", "shipments", "transportation", "haulage", "logistics"],
"passenger": ["travelers", "commuters", "tourists", "riders", "patrons"],
"travel": ["tourism", "trips", "vacations", "journeys", "expeditions"],
"postal": ["mail", "courier", "shipping", "delivery", "logistics"],
"intangible assets": ["intellectual property", "patents", "copyrights", "trademarks", "goodwill"],
"courier": ["messenger", "delivery service", "package carrier"],
"telecommunication": ["communication", "information technology", "networking"],
"satellite": ["space-based", "orbital", "extraterrestrial"],
"construction": ["building", "infrastructure", "development"],
"insurance": ["risk management", "indemnity", "underwriting"],
"finance": ["banking", "investment", "accounting"],
"information technology": ["computing", "software", "digital services"],
"intellectual property": ["patents", "copyrights", "trademarks"],
"trade": ["commerce", "business", "transactions"],
"legal": ["law", "judiciary", "compliance"],
"consulting": ["advisory", "guidance", "expertise"],
"research and development": ["innovation", "experimentation", "discovery"],
"compensation": ["wages", "salaries", "remuneration"],
"repatriation": ["repatriation", "restitution", "reimbursement"],
"export": ["overseas sales", "international trade", "foreign markets"],
"shipping": ["transport", "logistics", "freight"],
"travel": ["tourism", "hospitality", "leisure"],
"receipts": ["payments", "income", "revenue", "earnings", "proceeds"],
 "insurance": ["coverage", "protection", "indemnity", "assurance"],
 "premiums": ["fees", "charges", "rates", "costs"],
 "claims": ["demands", "requests", "applications", "assertions"],
 "financial": ["monetary", "economic", "fiscal"],
 "investment": ["capital", "funding", "financing"],
 "banking": ["financial services", "banking services", "banking operations"],
 "consultancy": ["advisory", "consulting", "professional services"],
 "software": ["applications", "programs", "systems"],
 "information": ["data", "intelligence", "knowledge"],
 "services": ["activities", "work", "operations", "functions"],
 "trade": ["commerce", "business", "transactions"],
 "legal": ["judicial", "law", "legislative"],
 "accounting": ["bookkeeping", "auditing", "taxation"],
 "management": ["administration", "operations", "organization"],
 "research": ["investigation", "analysis", "exploration"],
 "development": ["growth", "advancement", "progress"],
 "architectural": ["design", "planning", "construction"],
 "agricultural": ["farming", "cultivation", "horticulture"],
 "mining": ["extraction", "excavation", "prospecting"],
 "processing": ["manufacturing", "production", "refining"],
 "distribution": ["logistics", "supply chain", "delivery"],
 "environmental": ["ecological", "green", "sustainable"],
 "audio-visual": ["multimedia", "film", "entertainment"],
 "cultural": ["artistic", "heritage", "educational"],
 "maintenance": ["upkeep", "repair", "support"],
 "compensation": ["remuneration", "salary", "wages"],
 "dividends": ["returns", "payouts", "distributions"],
 "refunds": ["reimbursements", "returns", "repayments"]

}

class BM25Encoder:
    def __init__(self, corpus: List[str]):
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def encode_documents(self, documents: List[str]) -> List[Dict[str, List[float]]]:
        return [
            {"values": self.bm25.get_scores(doc.split()).tolist()}
            for doc in documents
        ]

# Initialize BM25Encoder with a sample corpus
sample_corpus = [
    "Trusty Money is a cross-border payment platform for businesses, offering services like currency conversions, local collections, remittances, and invoicing solutions.",
    "Currently available for businesses, launching soon for freelancers."
]
bm25_encoder = BM25Encoder(sample_corpus)

def expand_query_with_synonyms(query: str) -> str:
    expanded_query = query
    words = query.split()

    for word in words:
        synonyms = set()

        # Use custom synonym dictionary
        if word.lower() in custom_synonyms:
            synonyms.update(custom_synonyms[word.lower()])

        # If synonyms are found, append them to the expanded query
        if synonyms:
            expanded_query += " " + " ".join(synonyms)

    return expanded_query

def retrieve_documents(query: str) -> Optional[Dict[str, Any]]:
    expanded_query = expand_query_with_synonyms(query)
    dense_vector = embedding_model.encode(expanded_query).tolist()
    sparse_vector = bm25_encoder.encode_documents([expanded_query])[0]

    response = index.query(
        vector=dense_vector,
        top_k=5,
        include_values=False,
        include_metadata=True,
    )

    matches = []
    if response['matches']:
        for match in response['matches']:
            question = match['metadata']['question']
            answer = match['metadata']['answer']
            matches.append({'question': question, 'answer': answer, 'score': match['score']})

        matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = matches[0]
        confidence = "High Confidence" if best_match['score'] >= 0.95 else "Medium Confidence" if best_match['score'] >= 0.65 else "Low Confidence"

        return {'question': best_match['question'], 'answer': best_match['answer'], 'score': best_match['score'], 'confidence': confidence}

    return None

def refine_answer(query: str, initial_answer: str, confidence: str) -> str:
    model = genai.GenerativeModel('gemini-pro')

    refine_prompt = f"""
    You are assisting in answering questions related to Trusty Money's fintech services.
    The user asked: "{query}". The following answer was retrieved with a confidence level of {confidence}:
    Initial Answer: "{initial_answer}"

    Please refine the answer, considering these guidelines:
    1. Ensure the answer is accurate and concise.
    2. Do not introduce any new information that is not in the original answer.
    3. Improve clarity where necessary while retaining the answer's core meaning.
    4. Use a professional tone and language.

    Refined Answer:
    """

    try:
        response = model.generate_content(refine_prompt)
        refined_answer = response.text if hasattr(response, 'text') else str(response)
        logging.info(f"Refined answer generated: {refined_answer.strip()}")
        return refined_answer.strip()
    except Exception as e:
        logging.error(f"Error in refining answer: {str(e)}")
        return f"Error in refinement. Original answer: {initial_answer}"

@app.route('/chatbot', methods=['GET', 'POST'])

def chatbot():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    document = retrieve_documents(query)

    if document:
        initial_answer = document['answer']
        confidence = document['confidence']
        refined_answer = refine_answer(query, initial_answer, confidence)
        return jsonify({"confidence": confidence, "response": refined_answer})

    return jsonify({"confidence": "Low Confidence", "response": "No matching documents found."})

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    app.run(debug=True)

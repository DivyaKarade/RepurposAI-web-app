#/usr/bin/env python3

import requests
from typing import List, Optional, Dict
from ratelimit import limits, sleep_and_retry
import ml_model  
import statistics
def fetch_target_data(compound: str) -> Dict[str, object]:
    targets = ml_model.predict_targets(smiles=compound)
    # targets should be a dictionary, for example: {"gene": "TP53"}
    return targets 
    # return {"gene": "TP53"}

# ------------------------------
# Open Targets
# ------------------------------

class OpenTargetsClient:
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.session = requests.Session()

    def run_query(self, query: str, variables: Optional[dict] = None) -> dict:
        response = self.session.post(
            self.BASE_URL,
            json={"query": query, "variables": variables},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise ValueError(f"Open Targets API error {response.status_code}: {response.text}")
        data = response.json()
        if "errors" in data:
            raise ValueError(f"GraphQL error: {data['errors']}")
        return data

    def get_ensembl_id(self, gene_symbol: str) -> Optional[str]:
        query = """
        query targetSearch($queryStr: String!) {
          search(queryString: $queryStr, entityNames: ["target"]) {
            hits {
              id
              name
              score
            }
          }
        }
        """
        variables = {"queryStr": gene_symbol}
        data = self.run_query(query, variables)
        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if hits:
            return hits[0]["id"]  # e.g., "ENSG00000141510"
        return None

    def get_associated_diseases(self, ensembl_id: str, score_threshold: float = 0.3) -> List[Dict[str, str]]:
        query = """
        query associatedDiseases($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            associatedDiseases {
              rows {
                disease {
                  id
                  name
                }
                datasourceScores {
                  id
                  score
                }
              }
            }
          }
        }
        """
        data = self.run_query(query, {"ensemblId": ensembl_id})
        print(data)
        rows = data["data"]["target"]["associatedDiseases"]["rows"]
        diseases = []
        for row in rows:
            disease_id = row["disease"]["id"]
            disease_name = row["disease"]["name"]
            avg_score = sum(ds["score"] for ds in row["datasourceScores"]) / len(row["datasourceScores"])
            if avg_score >= score_threshold:
                diseases.append({"id": disease_id, "name": disease_name})
        return diseases
    def get_target_evidences(self, ensembl_id: str, efo_ids: Optional[List[str]] = None) -> List[dict]:

        query = """
        query evidenceForTarget($ensemblId: String!, $efoIds:[String!]!) {
        target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            evidences(efoIds: $efoIds) {
            count
            rows {
                disease { id name }
                resourceScore
            }
            }
        }
        }
        """
        variables = {
            "ensemblId": ensembl_id,
            "efoIds": efo_ids,
            "datasourceIds": ["intogen", "eva", "gene2phenotype"]
        }
        data = self.run_query(query, variables)
        rows = data.get("data", {}).get("evidence", {}).get("rows", [])
        return rows or []

    def summarize_evidence(self, evidences: List[dict]) -> Dict[str, object]:
        """Summarize Open Targets evidence list."""
        if not evidences:
            return {"count": 0, "avg_score": 0, "top_diseases": []}

        scores = [ev.get("resourceScore", 0) for ev in evidences if ev.get("resourceScore") is not None]
        avg_score = statistics.mean(scores) if scores else 0

        diseases = {}
        for ev in evidences:
            d = ev.get("disease", {}).get("name")
            if d:
                diseases[d] = diseases.get(d, 0) + 1

        top_diseases = sorted(diseases.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "count": len(evidences),
            "avg_score": avg_score,
            "top_diseases": [d for d, _ in top_diseases],
        }

# ------------------------------
# KEGG
# ------------------------------

class KEGGAPIClient:
    BASE_URL = "https://rest.kegg.jp"
    RATE_LIMIT_CALLS = 3
    RATE_LIMIT_PERIOD = 1

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.session = requests.Session()

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    def _make_request(self, endpoint: str) -> requests.Response:
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, timeout=self.timeout)
        self._validate_response(response)
        return response

    def _validate_response(self, response: requests.Response):
        if response.status_code != 200:
            raise ValueError(f"KEGG error {response.status_code}: {response.text}")

    def find_kegg_id(self, gene_symbol: str, organism: str = "hsa") -> Optional[str]:
        """Find KEGG gene ID."""
        response = self._make_request(f"find/genes/{gene_symbol}")
        for line in response.text.strip().splitlines():
            if line.startswith(f"{organism}:"):
                parts = line.split("\t")
                gene_id, desc = parts

                # Need to check whether the gene name actually matches with the gene symbol
                # e.g. first result for BRCA1 is NBR1 (hsa:4077), NOT BRCA1 (hsa:672)
                gene_names = desc.split(";")[0]
                primary_name = gene_names.split(",")[0].strip().upper()
                if primary_name == gene_symbol.upper():
                    return gene_id  # e.g. "hsa:672"
        return None

    def fetch_pathway_all(self, kegg_id: str) -> List[str]:
        """Fetch all pathways linked to a KEGG gene ID."""
        response = self._make_request(f"link/pathway/{kegg_id}")
        return [l.split()[1].replace("path:", "") for l in response.text.strip().splitlines() if "path:" in l]

# ------------------------------
# Main pipeline
# ------------------------------
if __name__ == "__main__":
    ## testing
    compound = "CC(=O)Oc1ccccc1C(=O)O " # Aspirin
    predicted_data = fetch_target_data(compound)
    print(predicted_data)
    gene_symbol = predicted_data[0]["gene"]

    print(f"Checking evidence for {gene_symbol} on Open Targets...")
    ot_client = OpenTargetsClient()

    # Step 1. Get Ensembl ID
    ensembl_id = ot_client.get_ensembl_id(gene_symbol)
    diseases = ot_client.get_associated_diseases(ensembl_id, score_threshold=0.5)
    # diseases = ot_client.get_efo_ids(diseases)
    efo_ids = [d["id"] for d in diseases]
    if not ensembl_id:
        print(f"No Ensembl ID found for {gene_symbol}, skipping.")
        exit(1)
    print(f"Ensembl ID: {ensembl_id}")

    evidences = ot_client.get_target_evidences(ensembl_id, efo_ids)
    summary = ot_client.summarize_evidence(evidences)

    if summary["count"] > 0 and summary["avg_score"] > 1e-20:
        kegg_client = KEGGAPIClient()
        kegg_id = kegg_client.find_kegg_id(gene_symbol)
        print(f"KEGG ID: {kegg_id}")

        if kegg_id:
            pathways = kegg_client.fetch_pathway_all(kegg_id)
            print(f"Pathways for {gene_symbol}: {pathways}")
        else:
            print("No KEGG ID found for this gene.")
    else:
        print(f"Skipping {gene_symbol}: insufficient evidence (count={summary['count']}, score={summary['avg_score']:.2e})")

import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import httpx
from groq import Groq


CWE_COLUMNS = [
    "cwe_CWE-20", "cwe_CWE-502", "cwe_CWE-74", "cwe_CWE-79",
    "cwe_CWE-862", "cwe_CWE-89", "cwe_NVD-CWE-noinfo", "cwe_other", "cwe_unknown"
]

MOTS_CRITIQUES = ["critical", "zero-day", "0-day", "actively exploited", "rce",
                  "remote code execution", "unauthenticated", "wild", "ransomware",
                  "data breach", "exploited"]

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


class CriticalityClassifier:
    model_name = "criticality_classifier"

    def __init__(self):
        artifacts_dir = Path(__file__).parent / "artifacts"

        with open(artifacts_dir / "xgb_pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        with open(artifacts_dir / "pca_embeddings.pkl", "rb") as f:
            self.pca = pickle.load(f)
        with open(artifacts_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.groq_client = None  # initialisation différée (lazy) : la clé n'est pas disponible au moment du build Docker
        self.nvd_api_key = os.environ.get("NVD_API_KEY")  # optionnel mais recommandé

        # En dev (mode natif, core hors Docker) : http://host.docker.internal:5001
        # En prod (core dans le même réseau Docker) : à adapter avec Fedi, ex. http://core:5000
        self.taranis_core_url = os.environ.get("TARANIS_CORE_URL", "http://host.docker.internal:5001")
        # TODO temporaire : token admin manuel pour les tests. En production, remplacer par
        # un BOT_API_KEY dédié généré par Taranis (comme les autres bots existants) — à valider avec Fedi.
        self.taranis_api_key = os.environ.get("TARANIS_API_KEY")

        self.urgence_map = {"Critique": "immediat", "Élevée": "sous_24h", "Faible": "planifie"}

        self._kev_ids_cache = None  # chargé au premier appel, pas au démarrage (évite un appel réseau bloquant à l'init)

    # ---------- Enrichissement CVE / CWE / KEV ----------

    async def _get_kev_ids(self) -> set:
        if self._kev_ids_cache is not None:
            return self._kev_ids_cache
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(CISA_KEV_URL)
                resp.raise_for_status()
                data = resp.json()
                self._kev_ids_cache = {v["cveID"] for v in data.get("vulnerabilities", [])}
        except Exception:
            self._kev_ids_cache = set()  # dégrade proprement plutôt que de planter
        return self._kev_ids_cache

    async def _fetch_cve_info(self, cve_id: str, client: httpx.AsyncClient) -> dict:
        headers = {"apiKey": self.nvd_api_key} if self.nvd_api_key else {}
        try:
            resp = await client.get(NVD_API_URL, params={"cveId": cve_id}, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                return {}
            cve_data = vulns[0]["cve"]

            cvss = None
            metrics = cve_data.get("metrics", {})
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    cvss = metrics[key][0]["cvssData"]["baseScore"]
                    break

            cwe = None
            for weakness in cve_data.get("weaknesses", []):
                for desc in weakness.get("description", []):
                    if desc.get("lang") == "en":
                        cwe = desc.get("value")
                        break
                if cwe:
                    break

            return {"cvss": cvss, "cwe": cwe}
        except Exception:
            return {}

    async def _enrich_cves(self, cve_ids: list[str]) -> dict:
        """Retourne max_cvss, cwe_principal, en interrogeant NVD pour chaque CVE."""
        if not cve_ids:
            return {"max_cvss": None, "cwe_principal": "unknown"}

        async with httpx.AsyncClient() as client:
            infos = [await self._fetch_cve_info(cve, client) for cve in cve_ids]

        cvss_scores = [i["cvss"] for i in infos if i.get("cvss") is not None]
        cwes = [i["cwe"] for i in infos if i.get("cwe")]

        return {
            "max_cvss": max(cvss_scores) if cvss_scores else None,
            "cwe_principal": cwes[0] if cwes else "unknown",
        }

    # ---------- Construction des 15 features tabulaires ----------

    async def _build_features(self, title: str, cve_ids: list[str], nb_domains: int, nb_urls: int, nb_ipv4s: int) -> np.ndarray:
        title_lower = title.lower()

        enrichment = await self._enrich_cves(cve_ids)
        kev_ids = await self._get_kev_ids()
        has_kev = any(cve in kev_ids for cve in cve_ids)

        title_length = len(title)
        title_word_count = len(title.split())
        kw_critical_count = sum(m in title_lower for m in MOTS_CRITIQUES)
        has_critical_keyword = int(kw_critical_count > 0)

        # IOC comptabilisés depuis les tags Taranis (déjà extraits par le bot IOC en amont),
        # cohérent avec la méthodologie du notebook d'entraînement (nb_domains/nb_urls/nb_ipv4s)
        nb_iocs_total = nb_domains + nb_urls + nb_ipv4s
        has_ioc = int(nb_iocs_total > 0)

        cwe_vec = [0] * len(CWE_COLUMNS)
        cwe_col = f"cwe_{enrichment['cwe_principal']}"
        if cwe_col in CWE_COLUMNS:
            cwe_vec[CWE_COLUMNS.index(cwe_col)] = 1
        elif enrichment["cwe_principal"] == "unknown":
            cwe_vec[CWE_COLUMNS.index("cwe_unknown")] = 1
        else:
            cwe_vec[CWE_COLUMNS.index("cwe_other")] = 1

        tabular = [
            title_length, title_word_count, kw_critical_count, has_critical_keyword,
            has_ioc, nb_iocs_total, *cwe_vec
        ]

        return np.array(tabular, dtype=float), enrichment["max_cvss"], has_kev, len(cve_ids)

    def _get_groq_client(self) -> Groq:
        if self.groq_client is None:
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        return self.groq_client

    def _calculer_signal_vigilance(self, max_cvss, has_kev) -> str:
        if has_kev or (max_cvss is not None and max_cvss >= 7):
            return "fort"
        if max_cvss is not None and 4 <= max_cvss < 7:
            return "modéré"
        return "faible"

    def _construire_prompt(self, title, niveau_predit, max_cvss, has_kev, nb_cve) -> str:
        cvss_txt = f"{max_cvss}" if max_cvss is not None else "non disponible (aucune CVE scorée identifiée)"
        return f"""Tu es un analyste senior en cybersécurité, spécialisé dans le triage de menaces (threat intelligence).

Voici une story de veille cyber déjà classée par un modèle de machine learning entraîné sur un score de criticité pondéré (50% CVSS, 30% présence CISA KEV, 10% nombre de CVE, 10% récence). Ta tâche : produire une explication et une recommandation d'action pour un analyste humain, à partir STRICTEMENT des faits fournis ci-dessous.

DONNÉES DE LA STORY :
- Titre : {title}
- Niveau de criticité prédit par le modèle : {niveau_predit}
- Score CVSS maximal : {cvss_txt}
- Présence dans la CISA KEV (exploitation active confirmée) : {"Oui" if has_kev else "Non"}
- Nombre de CVE associées : {nb_cve}

RÈGLES STRICTES :
1. N'invente AUCUN détail technique (version de logiciel, CVE, paramètre, méthode d'exploitation) qui ne figure pas explicitement dans le titre ou les données ci-dessus.
2. Reste factuel et concis. Rédige en français.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, respectant exactement ce format :
{{
  "explication": "résumé concis (2-3 phrases) justifiant le niveau de criticité prédit",
  "action_recommandee": "action concrète et réaliste, sans invention de détails techniques absents des données fournies",
  "note_analyste": "note libre signalant une éventuelle tension entre les faits chiffrés et le niveau prédit (vide si cohérent)"
}}"""

    async def predict(self, **kwargs) -> dict:
        title = kwargs["title"]
        tags = kwargs.get("tags", [])
        news_items = kwargs.get("news_items", [])
        if not tags and news_items:
            tags = [t for item in news_items for t in item.get("tags", [])]

        cve_ids = list(dict.fromkeys(
            t["name"] for t in tags if t["tag_type"] == "cves"
        ))
        nb_domains = sum(1 for t in tags if t["tag_type"] == "domains")
        nb_urls = sum(1 for t in tags if t["tag_type"] == "urls")
        nb_ipv4s = sum(1 for t in tags if t["tag_type"] == "ipv4s")

        tabular_features, max_cvss, has_kev, nb_cve = await self._build_features(
            title, cve_ids, nb_domains, nb_urls, nb_ipv4s
        )

        embedding = self.embed_model.encode([title])
        embedding_reduit = self.pca.transform(embedding)[0]

        X = np.concatenate([tabular_features, embedding_reduit]).reshape(1, -1)

        pred_encoded = self.pipeline.predict(X)
        niveau_predit = self.label_encoder.inverse_transform(pred_encoded)[0]
        urgence = self.urgence_map[niveau_predit]

        prompt = self._construire_prompt(title, niveau_predit, max_cvss, has_kev, nb_cve)
        completion = self._get_groq_client().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        llm_result = json.loads(completion.choices[0].message.content)

        return {
            "niveau_predit": niveau_predit,
            "niveau_urgence": urgence,
            "explication": llm_result["explication"],
            "action_recommandee": llm_result["action_recommandee"],
            "note_analyste": llm_result["note_analyste"],
            "signal_vigilance": self._calculer_signal_vigilance(max_cvss, has_kev),
        }

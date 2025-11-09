"""Job Search Agent - Job board integrations."""

import os
from utils.llm_providers import SESSION


class JobAgent:
    """Handles job search across multiple platforms."""
    
    def __init__(self, jooble_api_key: str | None = None):
        # Adzuna API credentials
        self.app_id = "aea2688c"
        self.app_key = "3d681c98182447e843823a9c9c2d14ee"
        self.base_url = "https://api.adzuna.com/v1/api/jobs"
        # Jooble API key
        self.jooble_api_key = jooble_api_key or os.getenv("JOOBLE_API_KEY")

    def _search_adzuna(self, query: str, location: str | None, num_results: int, 
                      country: str = "in", experience: int | None = None):
        """Search Adzuna job board."""
        url = f"{self.base_url}/{country}/search/1"
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": num_results,
            "what": query
        }
        if location:
            params["where"] = location
        if experience:
            params["experience"] = str(experience)
        
        response = SESSION.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        jobs = []
        for job in (data.get("results") or [])[:num_results]:
            jobs.append({
                "title": job.get("title"),
                "company": (job.get("company") or {}).get("display_name"),
                "location": (job.get("location") or {}).get("display_name"),
                "link": job.get("redirect_url")
            })
        return jobs

    def _search_jooble(self, query: str, location: str | None, num_results: int):
        """Search Jooble job board."""
        api_key = self.jooble_api_key or os.getenv("JOOBLE_API_KEY")
        if not api_key:
            return [{"error": "Jooble API key not set. Add your key in the sidebar and try again."}]
        
        url = f"https://jooble.org/api/{api_key}"
        payload = {
            "keywords": query or "",
            "page": 1,
        }
        if location:
            payload["location"] = location
        
        try:
            resp = SESSION.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("jobs") or data.get("results") or []
            
            jobs = []
            for job in items[:num_results]:
                jobs.append({
                    "title": job.get("title") or job.get("profession") or job.get("position"),
                    "company": job.get("company") or job.get("companyName") or job.get("employer"),
                    "location": job.get("location") or job.get("city") or job.get("country") or "",
                    "link": job.get("link") or job.get("url") or job.get("redirect_url")
                })
            return jobs if jobs else [{"error": "No jobs found"}]
        except Exception as e:
            return [{"error": f"Jooble error: {e}"}]

    def search_jobs(self, query, location=None, platform="adzuna", experience=None, 
                   num_results=10, country="in", jooble_api_key: str | None = None):
        """Search for jobs across platforms."""
        if jooble_api_key:
            self.jooble_api_key = jooble_api_key
        
        try:
            platform_norm = (platform or "adzuna").strip().lower()
            
            if platform_norm in ("adzuna",):
                jobs = self._search_adzuna(query, location, num_results, country=country, experience=experience)
                return jobs if jobs else [{"error": "No jobs found"}]
            elif platform_norm in ("jooble",):
                return self._search_jooble(query, location, num_results)
            else:
                return [{"error": f"Platform '{platform}' not supported. Choose Adzuna or Jooble."}]
        except Exception as e:
            return [{"error": str(e)}]

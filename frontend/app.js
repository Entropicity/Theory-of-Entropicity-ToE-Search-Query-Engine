const API_URL = "http://127.0.0.1:8000/search";

const queryInput = document.getElementById("query");
const searchBtn = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

async function performSearch() {
  const query = queryInput.value.trim();
  if (!query) {
    statusEl.textContent = "Please enter a question or phrase.";
    return;
  }

  statusEl.textContent = "Searching...";
  resultsEl.innerHTML = "";
  searchBtn.disabled = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: query,
        top_k: 5
      })
    });

    if (!response.ok) {
      statusEl.textContent = `Error: ${response.status} ${response.statusText}`;
      searchBtn.disabled = false;
      return;
    }

    const data = await response.json();
    const results = data.results || [];

    if (results.length === 0) {
      statusEl.textContent = "No results found.";
      searchBtn.disabled = false;
      return;
    }

    statusEl.textContent = `Found ${results.length} result(s).`;

    results.forEach((item, index) => {
      const div = document.createElement("div");
      div.className = "result-item";

      const meta = document.createElement("div");
      meta.className = "result-meta";
      meta.innerHTML = `
        <span>#${index + 1}</span>
        <span>Score: ${item.score.toFixed(3)}</span>
        <span>${item.source || ""}</span>
      `;

      const text = document.createElement("div");
      text.className = "result-text";
      text.textContent = item.text;

      div.appendChild(meta);
      div.appendChild(text);
      resultsEl.appendChild(div);
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error contacting the API. Is the server running?";
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener("click", performSearch);

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    performSearch();
  }
});


document.addEventListener("DOMContentLoaded", () => {
    const search = document.getElementById("history-search");
    if (search) {
        const rows = [...document.querySelectorAll("#history-table tbody tr")];
        search.addEventListener("input", () => {
            const query = search.value.toLowerCase().trim();
            rows.forEach(row => { row.hidden = !row.textContent.toLowerCase().includes(query); });
            document.getElementById("search-empty").hidden = rows.some(row => !row.hidden);
        });
    }
    const form = document.getElementById("predict-form");
    if (!form) return;
    const fields = ["TV", "Radio", "Newspaper"].map(name => document.getElementById(name));
    const total = document.getElementById("budget-total");
    const updateBudget = () => {
        if (!total) return;
        const sum = fields.reduce((sum, field) => {
            const value = Number(field.value);
            return sum + (Number.isFinite(value) && value >= 0 ? value : 0);
        }, 0);
        total.value = sum.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    };
    if (total) {
        fields.forEach(field => field.addEventListener("input", updateBudget));
        document.querySelectorAll("[data-budget]").forEach(button => {
            button.addEventListener("click", () => {
                const values = button.dataset.budget.split(",");
                fields.forEach((field, index) => { field.value = values[index]; });
                updateBudget();
            });
        });
        updateBudget();
    }
    form.addEventListener("submit", () => {
        if (!form.checkValidity()) return;
        const button = form.querySelector('button[type="submit"]');
        if (button) { button.disabled = true; button.textContent = "Estimating sales…"; }
    });
    window.addEventListener("pageshow", () => {
        const button = form.querySelector('button[type="submit"]');
        if (button) { button.disabled = false; button.textContent = "Predict Sales"; }
    });
});

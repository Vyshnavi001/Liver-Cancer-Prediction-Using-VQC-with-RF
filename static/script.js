document.getElementById("form").addEventListener("submit", function(e) {
    e.preventDefault();

    const data = {
        age: age.value,
        tb: tb.value,
        db: db.value,
        alk: alk.value,
        alt: alt.value,
        ast: ast.value,
        tp: tp.value,
        alb: alb.value,
        ratio: ratio.value
    };

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(res => {
        document.getElementById("result").innerHTML = `
            <h3>${res.result}</h3>
            <p><b>Confidence:</b> ${res.confidence}%</p>
            <p><b>Risk:</b> ${res.risk}</p>
            <p><b>Suggestions:</b><br>${res.suggestions.join("<br>")}</p>
        `;
    });
});
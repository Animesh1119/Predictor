document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    const formData = {
        stockName: document.getElementById("stockName").value,
        startDate: document.getElementById("startDate").value,
        endDate: document.getElementById("endDate").value,
        algorithm: document.getElementById("algorithm").value,
        learningRate: document.getElementById("learningRate").value,
        epochs: document.getElementById("epochs").value
    };

    console.log(formData);
    // You can add a call to your machine learning backend here.
});

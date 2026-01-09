// script.js - Add this to your GitHub repo
console.log("script.js loaded");

async function loadModel() {
    console.log("Loading model...");
    try {
        const model = await tf.loadLayersModel('model.json');
        console.log("Model loaded successfully!");
        console.log("Model input shape:", model.inputs[0].shape);
        console.log("Model output shape:", model.outputs[0].shape);
        return model;
    } catch (error) {
        console.error("Error loading model:", error);
        throw error;
    }
}

async function testModel() {
    try {
        const model = await loadModel();
        
        // Create a test image tensor
        const testTensor = tf.zeros([1, 224, 224, 3]);
        
        // Make prediction
        const prediction = model.predict(testTensor);
        const result = await prediction.data();
        
        console.log("Test prediction:", result);
        testTensor.dispose();
        prediction.dispose();
        
        return result;
    } catch (error) {
        console.error("Test failed:", error);
    }
}

// Expose functions to global scope
window.loadModel = loadModel;
window.testModel = testModel;

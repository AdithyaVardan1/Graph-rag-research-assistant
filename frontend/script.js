const API_URL = 'http://127.0.0.1:8000';
const buildBtn = document.getElementById('build-btn');
const topicInput = document.getElementById('topic-input');
const graphContainer = document.getElementById('graph-container');
const messageBox = document.getElementById('message-box');
const summarizeBtn = document.getElementById('summarize-btn');
const hypothesizeBtn = document.getElementById('hypothesize-btn');
const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const selectionInfo = document.getElementById('selection-info');
const outputContainer = document.getElementById('output-container');
const outputBox = document.getElementById('output-box');

let cy = null;

buildBtn.addEventListener('click', handleBuildGraph);
summarizeBtn.addEventListener('click', () => handleAgentAction('summarize'));
hypothesizeBtn.addEventListener('click', () => handleAgentAction('hypothesize'));
chatBtn.addEventListener('click', () => handleAgentAction('chat'));

async function handleBuildGraph() {
    const topic = topicInput.value.trim();
    if (!topic) return alert('Please enter a topic.');

    updateMessage('Building knowledge graph, please wait...');
    buildBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/api/build_graph`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, max_results: 20 })
        });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const graphData = await response.json();

        if (!graphData.nodes || graphData.nodes.length === 0) {
            updateMessage('No papers found for this topic.');
        } else {
            renderGraph(graphData);
        }
    } catch (error) {
        console.error("Critical error:", error);
        updateMessage(`An error occurred. Check the console (F12).`);
    } finally {
        buildBtn.disabled = false;
    }
}

async function handleAgentAction(action) {
    const selectedNodes = cy.$('node:selected');
    if (selectedNodes.length === 0) return alert("Please select papers first.");
    
    const selectedPapers = selectedNodes.map(node => node.data('fullData'));
    const query = chatInput.value.trim();

    if (action === 'chat' && !query) return alert("Please enter a question.");
    
    outputContainer.style.display = 'block';
    outputBox.textContent = `Processing request: ${action}...`;

    try {
        const response = await fetch(`${API_URL}/api/agent_action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, selected_papers: selectedPapers, query })
        });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        outputBox.textContent = data.response;
        if(action === 'chat') chatInput.value = '';
    } catch (error) {
        console.error("Agent action error:", error);
        outputBox.textContent = `An error occurred. Check the console.`;
    }
}

function renderGraph(graphData) {
    if (typeof cytoscape === 'undefined') {
        updateMessage("Error: Could not load the visualization library.");
        return;
    }

    try {
        messageBox.style.display = 'none';

        const elements = graphData.nodes.map(node => ({ group: 'nodes', data: { id: node.id, label: node.title, fullData: node }}));
        graphData.edges.filter(e => e.weight > 0.35).forEach(edge => {
            elements.push({ group: 'edges', data: { source: edge.source, target: edge.target }});
        });

        cy = cytoscape({
            container: graphContainer,
            elements,
            style: [
                { selector: 'node', style: { 'background-color': '#007bff', 'width': 20, 'height': 20, 'border-color': '#fff', 'border-width': 2 }},
                { selector: 'node[label]', style: { 'label': 'data(label)', 'font-size': '10px', 'color': '#444', 'text-valign': 'bottom', 'text-halign': 'center', 'text-margin-y': '5px', 'text-wrap': 'ellipsis', 'text-max-width': '120px' }},
                { selector: 'edge', style: { 'width': 1.5, 'line-color': '#cccccc', 'curve-style': 'bezier' }},
                { selector: 'node:selected', style: { 'background-color': '#ffc107', 'border-color': '#e0a800' } }
            ],
            layout: { name: 'cose', idealEdgeLength: 150, nodeRepulsion: 400000, fit: true, padding: 50, animate: true, animationDuration: 1000 },
            minZoom: 0.2, maxZoom: 3.0, wheelSensitivity: 0.1
        });

        cy.on('select unselect', 'node', () => {
            const selectedCount = cy.$('node:selected').length;
            const isDisabled = selectedCount === 0;
            
            summarizeBtn.disabled = isDisabled;
            hypothesizeBtn.disabled = isDisabled;
            chatInput.disabled = isDisabled;
            chatBtn.disabled = isDisabled;

            selectionInfo.textContent = isDisabled ? "Select papers to activate." : `${selectedCount} paper(s) selected.`;
        });

    } catch (error) {
        console.error("Failed to render graph:", error);
        updateMessage("An error occurred while displaying the graph.");
    }
}

function updateMessage(text) {
    messageBox.textContent = text;
    messageBox.style.display = 'flex';
}
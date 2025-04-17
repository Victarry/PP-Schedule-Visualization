// assets/clientside.js

// Make sure the assets folder is configured correctly in Dash for this to be loaded.
// Dash automatically serves files from a folder named 'assets' in the root directory.

if (!window.dash_clientside) { window.dash_clientside = {}; }

window.dash_clientside.clientside = {
    update_strategy_selection: function(n_clicks_all, current_selection) {
        // Determine which button triggered the callback
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            // Should not happen with prevent_initial_call=True, but handle defensively
            return dash_clientside.no_update;
        }

        const triggered_id_str = ctx.triggered[0].prop_id.split('.')[0];
        if (!triggered_id_str) {
            // If we can't parse the ID, don't update
            return dash_clientside.no_update;
        }

        // Parse the JSON ID string to get the actual index (strategy name)
        let triggered_index;
        try {
            const triggered_id_obj = JSON.parse(triggered_id_str);
            triggered_index = triggered_id_obj.index;
        } catch (e) {
            console.error("Error parsing callback context ID:", e);
            return dash_clientside.no_update; // Don't update if ID parsing fails
        }

        // --- Update Selection Logic ---
        // Initialize new_selection as a copy of the current selection
        let new_selection = current_selection ? [...current_selection] : [];

        // Toggle the selected state
        const index_in_selection = new_selection.indexOf(triggered_index);
        if (index_in_selection > -1) {
            // If already selected, remove it (allow deselecting all for now)
            new_selection.splice(index_in_selection, 1);
        } else {
            // If not selected, add it
            new_selection.push(triggered_index);
        }

        // --- Prepare Outputs ---
        const all_indices = ctx.inputs_list[0].map(input => input.id.index); // Get all strategy names from the Input IDs

        // Generate active states, colors, and outlines for ALL buttons
        const active_states = all_indices.map(index => new_selection.includes(index));
        const colors = active_states.map(active => active ? 'primary' : 'secondary'); // 'primary' for active, 'secondary' for inactive
        const outlines = active_states.map(active => !active); // Outline=true for inactive, false for active

        // Generate validation message
        const feedback = new_selection.length === 0 ? "Please select at least one strategy." : "";

        // Return updated store data, button states, and feedback
        return [new_selection, active_states, colors, outlines, feedback];
    }
    // Add other clientside functions here if needed
}; 
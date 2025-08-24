# Direct_Struct_Summary Folder Management Rules

## Purpose
The Direct_Struct_Summary folder contains JSON files that categorize project components by their relevance to the mini-planet generation system goals. This helps maintain project focus and identify components for cleanup.

## File Organization

### Core System Summaries
- `core_terrain_system.json` - Essential terrain generation components
- `rendering_pipeline.json` - OpenGL rendering and visualization 
- `build_system.json` - Build configuration and compilation
- `ai_mesh_building.json` - AI-driven mesh generation pipeline

### Secondary System Summaries  
- `agent_systems.json` - AI agent components (relevant vs legacy)
- `test_and_example_data.json` - Test data and example files

### Cleanup Guidance
- `irrelevant_components.json` - Components safe to delete

## Navigation Rules

### When Adding New Components
1. **Assess Relevance**: Does the component directly support:
   - Mini-planet generation using random seeds?
   - OpenGL rendering and navigation?
   - Build system functionality?
   - AI mesh-building loop system?

2. **Update Appropriate Summary**: Add to relevant JSON file under appropriate category

3. **Mark for Cleanup**: If irrelevant, add to `irrelevant_components.json`

### When Updating Existing Components
1. **Check Current Categorization**: Find component in existing summaries
2. **Reassess Relevance**: Has the component's role changed?
3. **Move Between Categories**: Update JSON files accordingly
4. **Update Dependencies**: Check if dependent components need recategorization

### When Removing Components
1. **Remove from Summaries**: Delete entries from all JSON files
2. **Check Dependencies**: Ensure no critical dependencies are broken
3. **Update Related Categories**: Adjust related component classifications

## Maintenance Workflow

### Weekly Review
1. Scan for new files in project root and major subdirectories
2. Update summaries with new components
3. Review `irrelevant_components.json` for safe deletion candidates
4. Validate that core system summaries are complete

### Major Refactoring
1. **Before**: Export current summaries as backup
2. **During**: Update summaries as components are moved/changed
3. **After**: Validate all summaries reflect new structure

### Deletion Process
1. **Review Dependencies**: Check `core_terrain_system.json` and `rendering_pipeline.json`
2. **Test Critical Paths**: Ensure mini-planet generation still works
3. **Remove from Summaries**: Delete entries from all JSON files
4. **Document Removal**: Note major deletions in project logs

## JSON File Format Rules

### Structure
```json
{
  "category_name": [
    "path/to/component1",
    "path/to/component2"
  ],
  "another_category": [
    "path/to/component3"
  ]
}
```

### Naming Conventions
- **Categories**: Use snake_case with descriptive names
- **Paths**: Use relative paths from project root
- **Files**: Group by functionality, not location

### Update Guidelines
- **Add**: Append to appropriate array
- **Remove**: Delete from all arrays
- **Move**: Remove from old category, add to new
- **Rename**: Update path strings consistently

## Critical Components (Never Delete)
- Core terrain generation (heightfield, noise, cubesphere)
- OpenGL viewers and rendering pipeline  
- Build system (CMakeLists.txt, Makefile, src/)
- Agent D terrain processing system
- Deterministic seeding and validation

## Safe to Delete Components
See `irrelevant_components.json` for current list including:
- Legacy evolution loops
- Unused AI tooling
- Non-essential schemas
- Demo and feedback systems not core to planet generation

## Integration with Development
- **Before Major Changes**: Review summaries to understand impact
- **During Development**: Update summaries as you work
- **Before Commits**: Ensure summaries reflect actual codebase state
- **Code Reviews**: Include summary updates in review scope
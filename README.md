# Deep Latent Concepts

This repository tries to explore the possibilities of multimodal models (CLIP) for finegrained object recognition tasks.


## Important Ideas
- **Concept**: Represents the semantic meaning for an object or an attribute such as "green", "dotted", "Bird", ...
- **Concept Space**: Is a collection of concepts that share the same abstract meaning but are disjoint classes such as "blue", "green", "yellow", ... representing concepts in the concept space of "color".
- **Concept Bank**: Is a collection of Concept Spaces that represent in total the complete capability to describe a target object.

#### Image feature pyramid
- In order to get a finegrained patchwise feature representation of an image using the CLIP architecture we will look into the usage of a feature pyramid. We split the image in patches of different sizes and pass them through the model to get different levels of resolution.
- The final feature representation is obtained by using the max operation.
  
## TODO
- [ ] Implement the object recognition downstream task on the CUBDataset
  - [ ] Extract relevant concepts from the attributes part in the dataset
  - [ ] Implement an evaluation
- [ ] Implement a GUI to manually adjust the property concepts of the target object
- [ ] Investigate other backbone models
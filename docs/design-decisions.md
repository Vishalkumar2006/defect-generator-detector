# Project design decisions

These decisions are project constraints unless they are changed through a documented experiment that records the hypothesis, protocol, evidence, and outcome.

## Dataset integrity

- Preserve the official KSDD2 test set without resplitting or using it for development decisions.
- Divide only the official training set into development training and validation subsets.
- Exclude the nonconforming exact copies `train\10301 (copy).png` and `train\10301_GT (copy).png` from every manifest, split, and loader. Do not delete or modify them.
- Require every future loader to consume a validated manifest. Unrestricted directory globbing is not an acceptable sample-discovery mechanism.
- Treat KSDD2 as a binary dataset: normal or defective. Do not infer defect-type labels.
- Keep original masks unchanged as the authoritative labels.

## Image geometry

- KSDD2 images are tall strips, approximately 230 pixels wide and 630 pixels high. Do not resize complete images into squares.
- Use reflection padding when a patch dimension is larger than the image width.
- Select patch size from measured mask bounding-box statistics. Candidate dimensions are expressed as width × height: 256 × 256, 256 × 384, and 256 × 512.
- Future detector testing must preserve native image geometry and use sliding-window inference if a whole image cannot be processed directly.

## Mask handling

- Ground-truth masks remain binary and use nearest-neighbour interpolation if a future operation requires resampling.
- Image and mask spatial transforms must be synchronized.
- Store derived dilated or feathered masks separately; they must never overwrite ground truth.
- Make positive random cropping mask-aware because many defects occupy only a very small image fraction.

## GAN direction derived from research

- Do not train a vanilla DCGAN, StyleGAN, or whole-image GAN directly on the limited defective set.
- Use a mask-guided residual-refinement GAN that preserves real normal images as backgrounds.
- Build coarse defect composites from training-only defect templates and normal development-training patches. The generator refines the local defect residual instead of regenerating the full image.
- Use a local PatchGAN discriminator and a strong background-preservation or self-regularization loss outside the defect mask.
- Apply lightweight differentiable discriminator augmentation to both real and fake inputs; never augment only real discriminator inputs.
- Begin with small translations, horizontal flips, and mild brightness/contrast changes.
- Avoid aggressive CutMix, MixUp, cutout, rotations, perspective transforms, and hue changes until a documented experiment justifies them.
- Preserve spatial annotations through every transform.
- Keep validation and test images outside GAN training, template extraction, and synthetic generation.
- Judge GAN usefulness primarily by improvement of a detector evaluated on real held-out images.
- Include nearest-neighbour and diversity checks to detect memorization.
- Record complete provenance metadata for every future synthetic image.


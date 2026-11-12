import * as THREE from 'three';

/**
 * N64 texture generation (Pokémon Stadium 2 `extras.sphereMap`): the base-colour map is looked up by the surface normal
 * in view space instead of the mesh UVs, so the image (a pre-lit sphere: Umbreon's dark gloss, Steelix's metal, Lugia's
 * pearl) stays put on screen while the model turns. The GLB's own UVs hold the same lookup frozen for a front view
 * (for viewers without this patch). Lighting still multiplies the result (the combiner's TEXEL0 x SHADE).
 */
export function installSphereMap(mat: THREE.MeshStandardMaterial): boolean {
  if (!mat.userData?.sphereMap || !mat.map || mat.userData.__sphereMap) return false;
  mat.userData.__sphereMap = true;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader, renderer) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <map_fragment>',
      `#ifdef USE_MAP
        vec3 smN = normalize( vNormal );
        vec4 sampledDiffuseColor = texture2D( map, vec2( smN.x, -smN.y ) * 0.5 + 0.5 );
        diffuseColor *= sampledDiffuseColor;
      #endif`,
    );
    prev?.call(mat, shader, renderer);
  };
  mat.needsUpdate = true;
  return true;
}

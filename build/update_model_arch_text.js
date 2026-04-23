import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const INPUT_PATH = path.resolve("../model_arch.pptx");
const OUTPUT_PATH = path.resolve("../model_arch_code_aligned.pptx");

const updates = [
  {
    slideIndex: 1,
    shapeIndex: 42,
    text: "Concat\n[p;c] (512d)\n↓\nMasked Pool\n↓\nClassifier",
  },
  {
    slideIndex: 3,
    shapeIndex: 23,
    text: "[EMA Update]           (α = 0.99, warmup 이후 iteration마다 갱신)",
  },
  {
    slideIndex: 3,
    shapeIndex: 32,
    text: "▸  Phase 1에서는 분류 헤드(Transformer, Classifier) 없이 GCN 표현 학습에만 집중하며, 초기 50 step은 distillation warmup으로 loss를 적용하지 않습니다.",
  },
  {
    slideIndex: 3,
    shapeIndex: 34,
    text: "▸  Teacher 파라미터는 역전파로 직접 업데이트되지 않고, warmup 이후 EMA를 통해서만 간접 업데이트됩니다.",
  },
  {
    slideIndex: 3,
    shapeIndex: 36,
    text: "▸  Student branch에는 edge dropout(p=0.1)이 적용되어 teacher보다 더 강한 regularization이 들어갑니다.",
  },
  {
    slideIndex: 4,
    shapeIndex: 4,
    text: "입력  (Teacher GCN 출력, no_grad)\n입력 LayerNorm + Dropout (p=0.3)",
  },
  {
    slideIndex: 4,
    shapeIndex: 42,
    text: "Post-norm block + 최종 LayerNorm",
  },
  {
    slideIndex: 4,
    shapeIndex: 50,
    text: "출력: 각 block에서 residual 이후 LayerNorm 적용\n최종 out_norm = LayerNorm(h) 후\npad_mask 위치는 0으로 마스킹 처리",
  },
  {
    slideIndex: 5,
    shapeIndex: 10,
    text: "Concat\n[h_p ; h_c]\n",
  },
];

const pptx = await FileBlob.load(INPUT_PATH);
const presentation = await PresentationFile.importPptx(pptx);

for (const update of updates) {
  const slide = presentation.slides.items[update.slideIndex];
  if (!slide) {
    throw new Error(`Missing slide at index ${update.slideIndex}`);
  }
  const shape = slide.shapes.items[update.shapeIndex];
  if (!shape) {
    throw new Error(
      `Missing shape at slide ${update.slideIndex + 1}, shape ${update.shapeIndex}`
    );
  }
  shape.text = update.text;
}

const output = await PresentationFile.exportPptx(presentation);
await output.save(OUTPUT_PATH);
console.log(`Saved ${OUTPUT_PATH}`);

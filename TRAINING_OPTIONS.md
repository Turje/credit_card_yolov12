# Training & Evaluation Options for Private Dataset

## Dataset Overview
- **query_images**: ~1000 original images (full dataset)
- **left_rotate**: Same images rotated left + zoomed (augmented)
- **right_rotate**: Same images rotated right + zoomed (augmented)
- All datasets: 16 categories annotated

---

## 🎯 Training Options

### **Option A: Combine All Three Datasets** ⭐ RECOMMENDED
**Strategy**: Merge query_images + left_rotate + right_rotate for training

**Pros:**
- ✅ Largest training set (~3000 images total)
- ✅ Better generalization (rotation + zoom augmentation)
- ✅ Model learns to handle rotated/zoomed objects
- ✅ More robust to real-world variations

**Cons:**
- ⚠️ Training takes longer (~3x data)
- ⚠️ May overfit if images are too similar (same base images)

**Implementation:**
- Use the merged dataset approach already in the notebook
- Train on combined dataset
- Validate/test on query_images split

---

### **Option B: Train on query_images Only**
**Strategy**: Use only original query_images for training

**Pros:**
- ✅ Fastest training
- ✅ Clean baseline model
- ✅ No data leakage concerns
- ✅ Standard evaluation protocol

**Cons:**
- ⚠️ Smaller dataset (~1000 images)
- ⚠️ May not generalize well to rotated/zoomed objects
- ⚠️ Misses opportunity to use augmented data

**Implementation:**
- Use query_images as ORIGINAL_DATASET
- Split into train/val/test
- Train normally

---

### **Option C: Train on Rotated Versions Only**
**Strategy**: Use left_rotate + right_rotate for training, query_images for test

**Pros:**
- ✅ Tests model robustness to rotation/zoom
- ✅ Simulates occlusion-like scenarios
- ✅ Good for testing generalization

**Cons:**
- ⚠️ Smaller training set (~2000 images)
- ⚠️ May not perform well on original orientation
- ⚠️ Less standard approach

**Implementation:**
- Merge left_rotate + right_rotate
- Use query_images as test set only

---

### **Option D: Train on query_images, Use Rotated as Occlusion Test**
**Strategy**: Train on query_images, evaluate on rotated versions as natural occlusion

**Pros:**
- ✅ Clean training (original images)
- ✅ Rotated versions serve as realistic occlusion test
- ✅ Tests model's ability to handle rotation/zoom

**Cons:**
- ⚠️ Rotated versions aren't true occlusion (just rotation)
- ⚠️ May need additional progressive occlusion tests

**Implementation:**
- Train on query_images split
- Evaluate on left_rotate + right_rotate as separate test sets
- Compare performance: original vs rotated

---

## 📊 Evaluation Options

### **Option 1: Progressive Occlusion (Crop Type)** ⭐ RECOMMENDED
**Strategy**: Generate 0%, 25%, 50%, 75% occlusion from test set using crop method

**Pros:**
- ✅ Systematic evaluation of occlusion impact
- ✅ Realistic (simulates camera panning/zooming)
- ✅ Clear performance degradation curves
- ✅ Identifies breaking points

**Implementation:**
```python
# After splitting query_images:
# 1. Generate progressive occlusion test sets
prepare_progressive_tests.py --test-dataset <test_split> --type crop

# 2. Evaluate on each occlusion level
evaluate_progressive.py --model <model_path> --test-sets <base_path>
```

**Output:**
- Performance metrics at each occlusion level
- Degradation curves (mAP vs occlusion %)
- Visualization of results

---

### **Option 2: Rotated Versions as Occlusion Test**
**Strategy**: Use left_rotate and right_rotate as natural occlusion tests

**Pros:**
- ✅ Already have the data (no generation needed)
- ✅ Tests rotation/zoom robustness
- ✅ Realistic scenario (objects at angles)

**Cons:**
- ⚠️ Not true progressive occlusion
- ⚠️ Hard to quantify occlusion level
- ⚠️ May not show gradual degradation

**Implementation:**
- Evaluate model on left_rotate and right_rotate separately
- Compare metrics: original vs left vs right
- Report performance drop

---

### **Option 3: Combined Evaluation** ⭐ BEST FOR COMPREHENSIVE ANALYSIS
**Strategy**: Both progressive occlusion + rotated versions

**Pros:**
- ✅ Comprehensive evaluation
- ✅ Tests both occlusion and rotation robustness
- ✅ Complete picture of model performance

**Implementation:**
1. Generate progressive occlusion (0%, 25%, 50%, 75%) from test set
2. Evaluate on progressive occlusion sets
3. Evaluate on left_rotate and right_rotate separately
4. Compare all results

**Output:**
- Progressive occlusion curves
- Rotation robustness metrics
- Combined analysis report

---

## 🎯 Recommended Approach

### **Training: Option A (Combine All Three)**
- Merge query_images + left_rotate + right_rotate
- Split merged dataset: 70% train, 15% val, 15% test
- Train YOLOv8 with pretrained weights
- **Reason**: Maximum data utilization, better generalization

### **Evaluation: Option 3 (Combined)**
- Generate progressive occlusion (0%, 25%, 50%, 75%) from test set
- Evaluate on progressive occlusion sets
- Evaluate on left_rotate and right_rotate as separate tests
- **Reason**: Comprehensive understanding of model robustness

---

## 📝 Implementation Steps

### Step 1: Merge Datasets
```python
# In notebook Cell 8, the code already merges datasets
# Ensure query_images, left_rotate, right_rotate are all detected
# The merged dataset will be created automatically
```

### Step 2: Split Merged Dataset
```python
# Cell 11: Split merged dataset
split_dataset.py --dataset <merged_dataset> --seed 42
```

### Step 3: Generate Progressive Occlusion Tests
```python
# Cell 12: Generate occlusion test sets from test split
prepare_progressive_tests.py --test-dataset <test_split> --type crop
```

### Step 4: Train Model
```python
# Cell 14-23: Train on merged dataset
# Uses pretrained weights from non-private model
```

### Step 5: Evaluate
```python
# Cell 24+: Evaluate on:
# 1. Progressive occlusion sets (0%, 25%, 50%, 75%)
# 2. left_rotate (as separate test)
# 3. right_rotate (as separate test)
```

---

## 🔄 Alternative: Separate Training Runs

If you want to compare approaches, you can train multiple models:

1. **Model A**: Train on query_images only → Evaluate on progressive occlusion
2. **Model B**: Train on merged dataset → Evaluate on progressive occlusion
3. **Model C**: Train on query_images → Evaluate on rotated versions

Compare which approach gives best occlusion robustness.

---

## 📈 Expected Results

### Progressive Occlusion (Crop Type)
- **0% occlusion**: Baseline performance (full objects visible)
- **25% occlusion**: Slight performance drop (~5-10%)
- **50% occlusion**: Moderate drop (~15-25%)
- **75% occlusion**: Significant drop (~30-50%)

### Rotated Versions
- **left_rotate**: Performance drop depends on rotation angle
- **right_rotate**: Performance drop depends on rotation angle
- Compare with original to measure rotation robustness

---

## ❓ Questions to Consider

1. **Primary Goal**: 
   - Maximize occlusion robustness? → Use Option A training + Option 3 evaluation
   - Baseline comparison? → Use Option B training + Option 1 evaluation

2. **Data Concerns**:
   - Are rotated versions too similar to originals? → Consider Option B
   - Want maximum data? → Use Option A

3. **Evaluation Focus**:
   - Systematic occlusion study? → Option 1
   - Rotation robustness? → Option 2
   - Comprehensive analysis? → Option 3

---

## 🚀 Quick Start Recommendation

**For your use case (private object detection with occlusion evaluation):**

1. **Training**: Option A (Combine all three datasets)
2. **Evaluation**: Option 3 (Progressive occlusion + rotated versions)
3. **Reason**: Maximum robustness testing, comprehensive evaluation

This gives you:
- Largest training set
- Systematic occlusion evaluation
- Rotation robustness testing
- Complete performance picture


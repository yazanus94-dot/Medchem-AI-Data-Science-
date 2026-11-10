# Feedback for SimResponder-B

96 responses graded: 55 correct, 14 partial, 27 incorrect (mean score 0.68).

## Prioritised issues

### 1. `contradicts_accepted_science` (4 responses, 4% of all)
Most affected questions: C03, C04, C05, C06

Highest priority. The answer asserts something that contradicts accepted practice. Add counter-example training items for these topics, require the response to state caveats, and review whether the model is repeating a common misconception.

### 2. `method_or_arithmetic_error` (14 responses, 15% of all)
Most affected questions: N04, N13, N17, N05, N06, N08

Wrong formula or arithmetic. Compare against the listed common mistaken approaches for the affected questions and add worked-solution examples that show the correct formula and why the tempting alternative fails.

### 3. `approximation_error` (7 responses, 7% of all)
Most affected questions: N04, N07, N10, N08

Close but outside tolerance, usually premature rounding. Encourage carrying full precision through intermediate steps and rounding only the final answer.

### 4. `power_of_ten_error` (5 responses, 5% of all)
Most affected questions: N18, N01, N09, N14

Factor-of-10^k errors point to unit-prefix slips. Add paired items that differ only by prefix and prompt a plausibility check (is this potency realistic for the assay range?).

### 5. `unit_missing` (4 responses, 4% of all)
Most affected questions: N05, N02, N03

The value is right but the unit is missing. Require a unit with every reported quantity in the answer format and in the reward signal.

### 6. `missing_most_key_points` (4 responses, 4% of all)
Most affected questions: C01, C02, C03

The explanation is too thin to be useful. Reward complete, structured answers that address every rubric point, not only a single correct sentence.

### 7. `missing_key_point` (3 responses, 3% of all)
Most affected questions: C03, C04, C05

Mostly right but one required point is missing. Use the rubric group coverage to see which point is dropped and add examples where that point is essential.

## Weakest topic areas

- Assay quality and statistics: mean score 0.25
- Assay preparation: mean score 0.50
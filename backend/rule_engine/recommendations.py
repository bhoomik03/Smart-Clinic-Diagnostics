"""
recommendations.py

Contains professional diet plans, home remedies (Desi Upchar), and exercise 
suggestions for various medical conditions detected by the AI rule engine.
"""

DISEASE_RECOMMENDATIONS = {
    "Diabetes": {
        "Diet": "• **Low Glycemic Index**: Prioritize Oats, Quinoa, and Brown Rice.\n• **High Fiber**: Aim for 30-40g fiber from leafy greens and legumes.\n• **Portion Control**: Use the 'Plate Method' (50% veggies, 25% protein, 25% complex carbs).\n• **Avoid**: Liquid calories, white bread, and high-sugar seasonal fruits.",
        "Remedies": "• **Fenugreek (Methi)**: Soak 2 tsp seeds overnight; consume with water on an empty stomach.\n• **Bitter Gourd (Karela)**: Drink 30ml fresh juice in the morning to improve insulin sensitivity.\n• **Cinnamon**: Add 1/2 tsp to tea or food to help manage glycemic response.\n• **Amla-Karela Blend**: A potent combination for long-term sugar regulation.",
        "Exercise": "• **Aerobic**: 30 mins of brisk walking or swimming (5 days/week).\n• **Resistance**: Light weight training or yoga twice a week to improve muscle glucose uptake.\n• **Yoga Poses**: Mandukasana (Frog Pose) and Paschimottanasana are highly recommended."
    },
    "Heart Disease": {
        "Diet": "• **DASH Diet**: Focus on fruits, vegetables, and low-fat dairy.\n• **Healthy Fats**: Increase Omega-3 via Walnut, Flaxseeds, and Olive Oil.\n• **Salt Reduction**: Limit sodium to < 1.5g per day; avoid processed deli meats.\n• **Soluble Fiber**: Oats and Beans to help lower 'Bad' LDL cholesterol.",
        "Remedies": "• **Garlic**: Consume 1-2 bruised raw cloves daily to support arterial health.\n• **Arjuna Bark (Arjun ki Chaal)**: Boil in milk/water for a traditional cardiac tonic.\n• **Bottlegourd (Lauki) Juice**: Excellent for heart health (ensure it is not bitter).\n• **Honey-Ginger**: A tsp for natural anti-inflammatory support.",
        "Exercise": "• **Steady Cardio**: 30 mins of walking, cycling, or light jogging.\n• **Avoid Static Loads**: No heavy bench press or straining without supervision.\n• **Cool-down**: Ensure 10 mins of slow walking and stretching after every session."
    },
    "Hypertension": {
        "Diet": "• **Potassium-Rich**: Bananas, Spinach, and Sweet Potatoes to balance sodium.\n• **Magnesium**: Seeds, nuts, and dark chocolate (>70%).\n• **Low Sodium**: Use herbs/spices like Lemon and Garlic instead of salt.\n• **Hydration**: Drink 2.5-3L water daily to maintain blood volume.",
        "Remedies": "• **Lemon Water**: Drink warm lemon water (no sugar/salt) first thing in the morning.\n• **Watermelon Seeds**: Dry and grind into a powder; consume 1 tsp daily.\n• **Tulsi (Holy Basil)**: Chew 4-5 leaves daily on an empty stomach.\n• **Hibiscus Tea**: Natural ACE-inhibitor properties to help lower BP.",
        "Exercise": "• **Dynamic Exercise**: Continuous movement like brisk walking or dancing.\n• **Pranayama**: Anulom Vilom (Alternate Nostril Breathing) for 15 mins daily.\n• **Meditation**: Mindfulness to lower sympathetic nervous system activity."
    },
    "Hypotension": {
        "Diet": "• **Salt Balance**: Slightly increase salt intake (e.g., Pink salt) in meals.\n• **Frequent Small Meals**: Prevents post-meal blood pressure drops.\n• **Caffeine**: Small amounts (Coffee/Tea) can help in acute low BP episodes.\n• **B-12 and Folate**: Critical for preventing anemia-linked hypotension.",
        "Remedies": "• **Raisins**: Soak 30-40 raisins overnight and eat them on an empty stomach.\n• **Licorice (Mulethi)**: Small amount of tea helps in raising blood pressure.\n• **Almond Milk**: Soak 7 almonds, peel, and blend with milk/saffron.\n• **Carrot-Beetroot Juice**: Daily intake to improve blood volume.",
        "Exercise": "• **Leg Exercises**: Squats or Calf raises to improve venous return to the heart.\n• **Gentle Yoga**: Focus on inversions only under expert guidance.\n• **Slow Transitions**: Always sit up slowly before standing."
    },
    "Obesity": {
        "Diet": "• **Protein First**: Ensure 0.8g-1g protein per kg body weight to maintain satiety.\n• **Low Carb**: Reduce sugar, starch, and refined flours.\n• **Volume Eating**: Fill up on salad and soup before the main course.\n• **Hidden Calories**: Avoid sauces, creams, and extra oils.",
        "Remedies": "• **Lemon-Honey Water**: Warm water with 1/2 lemon and 1 tsp honey on an empty stomach.\n• **Ginger-Cinnamon Tea**: Boosts thermogenesis and metabolic rate.\n• **Apple Cider Vinegar**: 1 tsp in a glass of water before heavy meals.\n• **Triphala Churna**: Take at bedtime to improve digestion and gut health.",
        "Exercise": "• **HIIT**: High-Intensity Interval Training (20 mins, 3 times/week).\n• **Compound Lifts**: Squats, deadlifts, or pushups for maximum calorie burn.\n• **Consistency**: Aim for 10,000 steps daily regardless of formal workouts."
    },
    "High Cholesterol": {
        "Diet": "• **Oatmeal**: Contains Beta-Glucan which binds cholesterol in the gut.\n• **Plant Sterols**: Fortified spreads or naturally in Nuts/Seeds.\n• **Fatty Fish**: Salmon/Mackerel for Omega-3 heart protection.\n• **Switch Oils**: Replace Butter/Lard with Olive or Sunflower oil.",
        "Remedies": "• **Coriander Seeds (Dhaniya)**: Boil 2 tsp in water, strain, and drink twice daily.\n• **Garlic**: Allicin in garlic helps reduce LDL and triglycerides.\n• **Isabgol (Psyllium Husk)**: 1 tsp in water before dinner for fiber binding.\n• **Green Tea**: Rich in catechins that help lower cholesterol absorption.",
        "Exercise": "• **Endurance**: Jogging, swimming, or brisk walking for over 30 mins.\n• **Frequency**: At least 150 mins of moderate-intensity activity weekly.\n• **Yoga**: Dhanurasana (Bow Pose) and Chakrasana for liver-gallbladder health."
    },
    "Kidney Disease": {
        "Diet": "• **Low Potassium**: Limit Banana, Potato, and Tomato; prefer Cauliflower/Berries.\n• **Low Phosphorus**: Avoid dark colas and highly processed dairy.\n• **Moderate Protein**: High-quality protein (Egg whites, lean fish) only.\n• **Fluid Management**: Follow your doctor's exact daily milliliter limit.",
        "Remedies": "• **Barley Water (Jau)**: Boiled barley water helps flush out minor toxins.\n• **Corn Silk Tea**: Traditional remedy for kidney and urinary support.\n• **Avoid NSAIDs**: Never take Ibuprofen or Naproxen as they damage kidneys.\n• **Ginger Tea**: May help manage nausea associated with renal issues.",
        "Exercise": "• **Light Stretching**: Keeps muscles active without metabolic overload.\n• **Walking**: 15-20 mins of slow walking in a cool environment.\n• **Qi Gong**: Gentle flowing movements to improve energy without stress."
    },
    "Blood Disorder (Haemogram)": {
        "Diet": "• **Heme Iron**: Lean meat and poultry (if non-veg).\n• **Non-Heme Iron**: Spinach, Lentils, Tofu, and Moringa (Drumstick leaves).\n• **Vit-C Buddy**: Always eat citrus with iron foods for 3x absorption.\n• **Inhibitors**: Avoid Tea/Coffee/Milk within 1 hour of iron-rich meals.",
        "Remedies": "• **Jaggery & Roasted Gram**: Traditional 'Ladoo' or snack for natural iron boost.\n• **Beetroot-Carrot Shot**: High in folic acid and iron (add a dash of lemon).\n• **Black Sesame Seeds**: Soak and consume 1 tsp daily.\n• **Cast Iron Cooking**: Use 'Loha Kadhai' for cooking to increase iron content.",
        "Exercise": "• **Fatigue Management**: Avoid high-intensity during low hb periods.\n• **Breathing Exercises**: Deep belly breathing to improve oxygen delivery.\n• **Slow Paced Walking**: To keep the lymphatic system moving."
    },
    "Dengue Fever": {
        "Diet": "• **Pure Hydration**: ORS, Coconut water, and fresh Orange juice.\n• **Soft Recovery Diet**: Kanji (Rice water), Khichdi, and clear soups.\n• **High Energy**: Dates and bananas to combat post-viral fatigue.\n• **Avoid**: Spicy, oily, and heavy fried foods during recovery.",
        "Remedies": "• **Papaya leaf**: 20ml juice once or twice daily (strongly consult doctor).\n• **Goat Milk**: Traditional belief to help with fever recovery (hygiene is key).\n• **Pomegranate**: Helps in improving blood count and energy levels.\n• **Giloy Decoction**: Known for building immunity during viral infections.",
        "Exercise": "• **Immune Rest**: No exercise during high fever; rest is mandatory.\n• **Active Recovery**: Gentle indoor walking only once platelets stabilize.\n• **Breathing**: Bhramari Pranayama (Bee Breath) for mental relaxation."
    },
    "Enteric Fever (Typhoid)": {
        "Diet": "• **Bland Diet**: White rice, toast, and boiled potatoes.\n• **Gut Rest**: Avoid high-fiber raw veggies; prefer well-cooked vegetable mash.\n• **Fluid Intake**: Sip warm water or mild herbal tea throughout the day.\n• **Probiotics**: Fresh curd or buttermilk to restore gut flora.",
        "Remedies": "• **Cloves (Laung)**: Boil 5-8 cloves in water and sip throughout the day.\n• **Tulsi-Ginger Juice**: 1 tsp for antimicrobial support.\n• **Honey**: Natural energy and antibacterial support for the gut.\n• **Apple Juice**: Provides quick energy and electrolytes.",
        "Exercise": "• **Strict Rest**: Typhoid causes intestinal weakness; avoid abdominal strain.\n• **Minor Movement**: Stretching in bed to prevent muscle stiffness.\n• **Post-Recovery**: Wait 2 weeks after fever ends before joining gym."
    },
    "Liver Dysfunction": {
        "Diet": "• **Cruciferous**: Broccoli and Cabbage to boost detox enzymes.\n• **Beetroot**: Contains Betalains that reduce liver inflammation.\n• **Coffee**: Research shows 1-2 black coffees can reduce liver scarring risk.\n• **Sugar/Fructose**: Limit high fructose corn syrup and table sugar.",
        "Remedies": "• **Turmeric (Haldi)**: Warm water with turmeric to reduce liver oxidative stress.\n• **Amla Juice**: Higher Vit-C helps in glutathione production.\n• **Sugarcane Juice**: Freshly prepared for jaundice/liver-related energy loss.\n• **Papaya Seeds**: Small amounts are traditionally used for liver health.",
        "Exercise": "• **Low-Impact Cardio**: Walking or swimming (increases blood flow to liver).\n• **Vajrasana**: The only pose allowed after meals to aid digestion.\n• **Avoid Alcohol**: Zero tolerance policy for liver recovery."
    },
    "Systemic Inflammation": {
        "Diet": "• **Omega-3s**: Walnuts, Flaxseeds, and small cold-water fish.\n• **Berries**: Blueberries, Raspberries, and Strawberries (High antioxidants).\n• **Turmeric & Pepper**: Piperine in pepper increases turmeric absorption by 2000%.\n• **Avoid**: Trans fats, Margarine, and highly refined seed oils.",
        "Remedies": "• **Golden Milk**: Milk with turmeric, cinnamon, and a pinch of black pepper.\n• **Green Tea**: 2-3 cups daily for EGCG (powerful antioxidant).\n• **Boswellia (Shallaki)**: Traditional herb for joint-related inflammation.\n• **Pineapple**: Contains Bromelain, a natural anti-inflammatory enzyme.",
        "Exercise": "• **Yoga**: Poses that decompress joints like Tadasana.\n• **Mobility Work**: 10 mins of joint rotations (Neck, Shoulder, Knee).\n• **Walking**: Low-stress rhythmic movement to clear cytokines."
    },
    "Cough/Cold/Flu": {
        "Diet": "• **Warmth**: Chicken soup or clear vegetable broth with extra garlic.\n• **Vit-C Boost**: Kiwis, Bell peppers, and Citrus fruits.\n• **Soothing**: Warm water with Honey and Ginger juice.\n• **Avoid Dairy**: If it increases mucus thickness for the individual.",
        "Remedies": "• **Saltwater Gargle**: 3 times a day to reduce throat inflammation.\n• **Steam**: Use hot water with a drop of Eucalyptus or Tea tree oil.\n• **Tulsi-Ginger Kadha**: Traditional decoction with black pepper and honey.\n• **Turmeric Gargle**: Antibacterial support for local throat infection.",
        "Exercise": "• **Check the Neck**: If symptoms are 'above the neck', light walking is okay.\n• **Full Rest**: If symptoms are 'below the neck' (Chest/Body), sleep is priority."
    }
}

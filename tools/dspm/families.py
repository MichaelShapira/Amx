"""S3 template families (design 24.3). Template 0 of every family is dev-only; the rest are
test-only, so no test phrasing is seen while the policy is tuned.

Channels: chat, email, genai and code_question feed W3 (inline DLP); email, memo, letter,
log, config and code-<lang> feed W1 (bulk scan) as chunk components. Families marked
filler=True are benign records used to pad chunks and chat threads. A template's
categories are the tags of its placeholders (<<ssn:GOV_ID>>), so a family may mix
categories; negative families must contain no tagged value.
"""

from __future__ import annotations

from tools.dspm.generator import Family

W3 = ("W3",)
W1 = ("W1",)
BOTH = ("W1", "W3")


def F(name, channel, kind, templates, workloads=BOTH, category=None, hard_case=None, filler=False):
    return Family(name, channel, kind, tuple(templates), tuple(workloads), category, hard_case, filler)


FAMILIES: list[Family] = [
    # ======================================================================= chat (W3)
    F("chat/hr-ssn", "chat", "pos", [
        "hey <<first_name2>>, payroll still needs my SSN for the W-4, it's <<ssn:GOV_ID>>. thanks!",
        "Quick one: the background check form wants my social, <<ssn:GOV_ID>>. Can you submit it for me?",
        "<<first_name2>> I'm locked out of the benefits portal. They asked for my SSN (<<ssn:GOV_ID>>), can you pass it to HR?",
        "For the I-9 you asked about: SSN <<ssn:GOV_ID>>, same as on my offer letter.",
    ], W3, category="GOV_ID"),
    F("chat/travel-id", "chat", "pos", [
        "For the visa invitation letter: passport number <<passport:GOV_ID>>, issued <<date>>.",
        "Booking the flights now. The airline profile needs my passport number, it's <<passport:GOV_ID>>.",
        "travel desk asked for my passport no. -> <<passport:GOV_ID>> (expires <<date>>)",
        "Can you add me as second driver on the rental? License <<drivers_license:GOV_ID>>, issued in <<city>>.",
        "UK payroll needs my NI number: <<nino:GOV_ID>>. Sorry it took a while!",
    ], W3, category="GOV_ID"),
    F("chat/ssn-keyword", "chat", "pos", [
        "SSN <<ssn_nodash:GOV_ID>> - please fix it in the employee record",
        "my social security number is <<ssn_nodash:GOV_ID>>, not the one on the form",
        "Tax filing question. SSN: <<ssn_nodash:GOV_ID>>, filing jointly this year.",
        "the bank says my social (<<ssn_nodash:GOV_ID>>) doesn't match, can HR check?",
    ], W3, category="GOV_ID", hard_case="keyword-proximity"),
    F("chat/contact-share", "chat", "pos", [
        "Hi, I'm <<full_name:PERSONAL>>. Best to reach me on my cell <<phone:PERSONAL>> or at <<personal_email:PERSONAL>> after 6pm.",
        "New tenant details for the lease: <<full_name:PERSONAL>>, <<home_address:PERSONAL>>, mobile <<phone:PERSONAL>>.",
        "Send the package to me at home please: <<full_name:PERSONAL>>, <<home_address:PERSONAL>>",
        "Emergency contact form: <<full_name:PERSONAL>>, personal email <<personal_email:PERSONAL>>, cell <<phone:PERSONAL>>.",
    ], W3, category="PERSONAL"),
    F("chat/dob-location", "chat", "pos", [
        "Insurance form needs: <<full_name:PERSONAL>>, date of birth <<dob:PERSONAL>>.",
        "<<full_name:PERSONAL>> (born <<dob:PERSONAL>>) is joining the trip, please add them to the manifest.",
        "Courier: <<full_name:PERSONAL>> is waiting at <<geo:PERSONAL>> right now, please hurry.",
        "Club registration for my kid: <<full_name:PERSONAL>>, DOB <<dob:PERSONAL>>, address <<home_address:PERSONAL>>.",
    ], W3, category="PERSONAL"),
    F("chat/bank-details", "chat", "pos", [
        "Hi team, my new bank details for payroll are account <<bank_account:FINANCIAL>>, routing <<routing:FINANCIAL>>.",
        "Changed banks, so direct deposit should go to routing <<routing:FINANCIAL>> / acct <<bank_account:FINANCIAL>> from next month.",
        "Please wire the deposit to IBAN <<iban:FINANCIAL>>, account holder <<full_name>>.",
        "refund can go straight to my account: <<iban:FINANCIAL>> (<<full_name>>)",
    ], W3, category="FINANCIAL"),
    F("chat/card-share", "chat", "pos", [
        "use my card for the team dinner: <<card:FINANCIAL>>, exp <<expiry>>, cvv <<cvv:FINANCIAL>>",
        "Can you book the hotel on my corporate card? <<card:FINANCIAL>> exp <<expiry>>",
        "card for the conference registration is <<card:FINANCIAL>>, security code <<cvv:FINANCIAL>>",
        "Just pay the vendor with <<card:FINANCIAL>>, I'll expense it later.",
    ], W3, category="FINANCIAL"),
    F("chat/health-share", "chat", "pos", [
        "FYI <<full_name:HEALTH>> won't be in this week, they were diagnosed with <<diagnosis:HEALTH>> and start <<medication:HEALTH>> tomorrow.",
        "Covering for <<full_name:HEALTH>>: the doctor says <<diagnosis:HEALTH>>, out until <<date>>.",
        "<<full_name:HEALTH>>'s insurance member ID is <<insurance_id:HEALTH>>, the clinic needs it before <<procedure:HEALTH>>.",
        "Lab results for MRN <<mrn:HEALTH>> (<<full_name:HEALTH>>) are back: <<diagnosis:HEALTH>> confirmed.",
    ], W3, category="HEALTH"),
    F("chat/password-share", "chat", "pos", [
        "the admin password for the staging dashboard is <<password:CREDENTIALS>>, don't share it",
        "wifi for the offsite: network <<product>>-Guest, password <<password:CREDENTIALS>>",
        "login for the vendor portal: <<username>> / <<password:CREDENTIALS>>",
        "I reset the shared mailbox, new password is <<password:CREDENTIALS>> until IT fixes SSO.",
    ], W3, category="CREDENTIALS"),
    F("chat/token-share", "chat", "pos", [
        "here's the slack bot token for the integration: <<slack_token:CREDENTIALS>>",
        "use my GitHub token <<github_pat:CREDENTIALS>> to pull the private repo for now",
        "Stripe key for the demo: <<stripe_key:CREDENTIALS>>",
        "the OpenAI key for the hackathon is <<openai_key:CREDENTIALS>>, please don't commit it",
        "pasting the bearer token so you can repro: <<jwt:CREDENTIALS>>",
    ], W3, category="CREDENTIALS"),
    F("chat/test-labeled-secret", "chat", "pos", [
        "This is just a test key, rotate it later: <<aws_key_id:CREDENTIALS>> / <<aws_secret:CREDENTIALS>>",
        "dummy creds for the sandbox (not prod!): user <<username>> pass <<password:CREDENTIALS>>",
        "sample token for the tutorial, it's a throwaway: <<github_pat:CREDENTIALS>>",
        "test-only stripe key, safe to share: <<stripe_key:CREDENTIALS>>",
    ], W3, category="CREDENTIALS", hard_case="test-labeled-secret"),
    F("chat/confidential", "chat", "pos", [
        "<<conf_marker:CONFIDENTIAL>>: <<deal:CONFIDENTIAL>>. Keep it quiet until the announcement.",
        "Heads up (<<conf_marker:CONFIDENTIAL>>): <<unreleased:CONFIDENTIAL>>.",
        "<<conf_marker:CONFIDENTIAL>> - roadmap for the board deck: <<roadmap:CONFIDENTIAL>>.",
        "sharing our top accounts list, <<conf_marker:CONFIDENTIAL>>: <<customer_list:CONFIDENTIAL>>",
    ], W3, category="CONFIDENTIAL"),
    # chat negatives
    F("chat/smalltalk", "chat", "neg", [
        "Thanks <<first_name>>, see you at standup.",
        "anyone up for lunch at 12:30? the new ramen place near the <<city>> office is open",
        "Running 5 min late to the <<team>> sync, start without me.",
        "Great demo today! The new dashboard looks much faster.",
        "Can someone review my PR before <<weekday>>? It's the one about retries in <<service>>.",
        "Reminder: the office is closed on <<date>> for the holiday.",
        "I'll be OOO <<weekday>> afternoon, ping <<first_name2>> if anything breaks.",
        "Deploy of <<service>> <<version>> finished, all checks green.",
        "Is the <<team>> retro still on for <<time>>?",
        "Nice work on the launch, everyone. Cake in the kitchen at 3.",
        "The VPN is slow again this morning, anyone else?",
        "I moved our 1:1 to <<weekday>>, hope that works.",
        "Who owns the flaky test in <<service>>? It failed twice today.",
        "Docs for <<product>> <<version>> are up on the wiki.",
        "Heads up: the <<city>> office printer is out of toner.",
        "Can we push the planning meeting by 15 minutes?",
        "Thanks for the quick review!",
        "The incident from last night is resolved, postmortem on <<weekday>>.",
        "Coffee machine on floor 2 is fixed.",
        "Reminder to submit your expense reports by <<date>>.",
    ], W3, filler=True),
    F("chat/order-number", "chat", "neg", [
        "Your order <<order_ref>> has shipped. Tracking number <<tracking>>.",
        "Where is invoice <<invoice_no>>? Accounts says it was paid on <<date>>.",
        "the failing build is <<gitsha>>, request id <<uuid>> if you want to check the logs",
        "Reference <<order_ref>> was refunded, the customer got an email.",
        "checksum of the release tarball: <<sha>>",
    ], W3, hard_case="look-alike-number"),
    F("chat/masked", "chat", "neg", [
        "Refund issued to the card ending in <<card_tail>>, the customer was notified.",
        "Their SSN on file is <<masked_ssn>>, so it matches the last four they gave.",
        "the payout went to <<masked_account>>, can you confirm with finance?",
        "I can only see <<masked_card>> in the CRM, the rest is hidden.",
    ], W3, hard_case="masked"),
    F("chat/doc-values", "chat", "neg", [
        "In staging just use the test card <<doc_card>> with any future date.",
        "The docs use <<doc_aws_key_id>> as the access key, replace it with your own.",
        "set API_KEY=YOUR_API_KEY in your .env, the README explains where to get one",
        "Use 4242 4242 4242 4242 for the checkout tests, it always succeeds.",
    ], W3, hard_case="documentation-value"),
    F("chat/topic-only", "chat", "neg", [
        "How often do we rotate API keys? Is it 90 days?",
        "Reminder: never paste passwords or tokens into Slack, use the vault.",
        "does anyone know the retention rule for SSNs in the HR system?",
        "HIPAA training is due Friday. Please don't email patient records.",
        "What's our process if a customer sends their card number in chat?",
    ], W3, hard_case="topic-only"),
    F("chat/first-name-only", "chat", "neg", [
        "<<first_name>> from accounting called about the <<product>> invoice.",
        "Ask <<first_name>> for the slides, she has the latest version.",
        "Welcome <<first_name>> to the <<team>> team!",
        "<<first_name>> and <<first_name2>> will run the workshop on <<weekday>>.",
    ], W3, hard_case="first-name-only"),
    F("chat/number-no-keyword", "chat", "neg", [
        "Ticket <<ninedigit>> is a duplicate, closing it.",
        "Part <<ninedigit>> is back-ordered until <<date>>.",
        "Build <<ninedigit>> passed on the second try.",
        "confirmation code <<ninedigit>> for the room booking",
    ], W3, hard_case="number-without-keyword"),
    F("chat/public-business", "chat", "neg", [
        "Saw the press release: <<company>> named <<exec_name>> as its new <<exec_title>>.",
        "<<company>> moved its headquarters to <<office_address>>, it was in the news today.",
        "Their CEO <<exec_name>> keynotes the conference on <<date>>, the agenda is public.",
    ], W3, hard_case="public-business"),
    F("chat/indirect-reference", "chat", "neg", [
        "The DB password is in the vault under secret/data/<<service>>/db, ask for access.",
        "Read the token from the GITHUB_TOKEN env var, don't hardcode it.",
        "Stripe keys are in 1Password, item '<<service>> prod'. Request access in #it-help.",
    ], W3, hard_case="indirect-reference"),

    # ======================================================================= email (W1 + W3)
    F("email/payroll-change", "email", "pos", [
        "Subject: Direct deposit update\n\nHi <<first_name2>>,\n\nPlease update my direct deposit. Routing number <<routing:FINANCIAL>>, account number <<bank_account:FINANCIAL>>.\n\nThanks,\n<<first_name>>",
        "Subject: New bank account for my salary\n\nHello payroll,\n\nFrom next month my salary should go to account <<bank_account:FINANCIAL>> (routing <<routing:FINANCIAL>>). The old account is closed.\n\nBest regards,\n<<first_name>>",
        "Subject: Supplier bank details\n\nDear accounts team,\n\nOur bank details changed. IBAN: <<iban:FINANCIAL>>, beneficiary <<company>>.\n\nKind regards,\n<<first_name>>",
    ], category="FINANCIAL"),
    F("email/card-payment", "email", "pos", [
        "Subject: Payment for order\n\nHi,\n\nPlease charge my card <<card:FINANCIAL>>, expiry <<expiry>>, CVV <<cvv:FINANCIAL>>, for the remaining balance of <<amount>>.\n\n<<first_name>>",
        "Subject: Re: invoice <<invoice_no>>\n\nYou can take the payment from my Visa <<card:FINANCIAL>> (exp <<expiry>>). Let me know when it's done.\n\nThanks, <<first_name>>",
        "Subject: Booking deposit\n\nHello,\n\nI'd like to pay the deposit by card: <<card:FINANCIAL>>, security code <<cvv:FINANCIAL>>.\n\nRegards,\n<<first_name>>",
    ], category="FINANCIAL"),
    F("email/gov-id", "email", "pos", [
        "Subject: Onboarding documents\n\nHi HR,\n\nFor my onboarding: SSN <<ssn:GOV_ID>>, driver's license <<drivers_license:GOV_ID>>.\n\nThanks,\n<<first_name>>",
        "Subject: W-9 for vendor setup\n\nHello,\n\nOur EIN is <<ein:GOV_ID>>. The signed W-9 is attached.\n\nRegards,\n<<first_name>>, <<company>>",
        "Subject: Visa paperwork\n\nDear <<first_name2>>,\n\nMy passport number is <<passport:GOV_ID>> and it is valid until <<date>>. Please add it to the invitation letter.\n\nBest,\n<<first_name>>",
        "Subject: Tax form\n\nHi,\n\nI don't have an SSN yet, so please use my ITIN <<itin:GOV_ID>> on the 1099.\n\nThanks,\n<<first_name>>",
    ], category="GOV_ID"),
    F("email/personal-record", "email", "pos", [
        "Subject: Address change\n\nHello,\n\nPlease update my records: <<full_name:PERSONAL>>, new home address <<home_address:PERSONAL>>, mobile <<phone:PERSONAL>>.\n\nThank you",
        "Subject: Reference check\n\nHi <<first_name2>>,\n\nThe candidate is <<full_name:PERSONAL>>, date of birth <<dob:PERSONAL>>, reachable at <<personal_email:PERSONAL>>.\n\nBest,\n<<first_name3>>",
        "Subject: Emergency contact\n\nHi HR,\n\nMy emergency contact is <<full_name2:PERSONAL>>, <<phone2:PERSONAL>>, <<home_address2:PERSONAL>>.\n\n<<first_name>>",
    ], category="PERSONAL"),
    F("email/health", "email", "pos", [
        "Subject: Sick leave\n\nHi <<first_name2>>,\n\n<<full_name:HEALTH>> has a confirmed <<diagnosis:HEALTH>> diagnosis and is on <<medication:HEALTH>>; the doctor's note says two weeks off.\n\n<<first_name3>>",
        "Subject: Referral\n\nDear Dr. <<last_name2>>,\n\nReferring <<full_name:HEALTH>> (MRN <<mrn:HEALTH>>) for <<procedure:HEALTH>>. History of <<diagnosis:HEALTH>>.\n\nRegards",
        "Subject: Claim follow-up\n\nHello,\n\nClaim for <<full_name:HEALTH>>, member ID <<insurance_id:HEALTH>>, for <<procedure:HEALTH>> on <<date>>. Please advise.\n\nThanks",
    ], category="HEALTH"),
    F("email/credentials", "email", "pos", [
        "Subject: Access to the reporting DB\n\nHi <<first_name2>>,\n\nConnection string: <<db_uri:CREDENTIALS>>\nPlease change the password after first login.\n\n<<first_name>>",
        "Subject: API access\n\nHello,\n\nYour API key for the partner sandbox is <<google_api_key:CREDENTIALS>>. Keep it secret.\n\nThe <<team>> team",
        "Subject: Temporary password\n\nHi <<first_name2>>,\n\nYour account <<username>> has a temporary password: <<password:CREDENTIALS>>\n\nIT Help Desk",
        "Subject: deploy key\n\nHere is the deploy key for the build server, the old one expired:\n\n<<pem_key:CREDENTIALS>>\n\n<<first_name>>",
    ], category="CREDENTIALS"),
    F("email/confidential", "email", "pos", [
        "Subject: <<conf_marker:CONFIDENTIAL>> - board update\n\nTeam,\n\n<<deal:CONFIDENTIAL>>. Please do not discuss this outside the deal team.\n\n<<first_name>>",
        "Subject: Q results preview\n\n<<conf_marker:CONFIDENTIAL>>\n\nAhead of the earnings call: <<unreleased:CONFIDENTIAL>>.\n\n<<first_name>>",
        "Subject: Roadmap draft\n\n<<conf_marker:CONFIDENTIAL>>\n\nFor leadership review: <<roadmap:CONFIDENTIAL>>.\n\nThanks,\n<<first_name>>",
    ], category="CONFIDENTIAL"),
    F("email/benign", "email", "neg", [
        "Subject: Team offsite\n\nHi all,\n\nThe offsite is on <<date>> in <<city>>. Agenda and travel details will follow next week.\n\n<<first_name>>",
        "Subject: Maintenance window\n\nHello,\n\n<<service>> will be unavailable on <<date>> from <<time>> for <<num>> minutes while we upgrade to <<version>>.\n\nThe <<team>> team",
        "Subject: Lunch and learn\n\nJoin us <<weekday>> at <<time>> for a talk on observability. Pizza provided!\n\n<<first_name>>",
        "Subject: Re: slides\n\nThanks <<first_name2>>, the slides look great. I moved the pricing section to the end.\n\n<<first_name>>",
        "Subject: Quarterly survey\n\nHi team,\n\nPlease fill in the engagement survey by <<date>>. It takes about 5 minutes.\n\nPeople Ops",
        "Subject: Welcome\n\nPlease welcome <<first_name>>, who joins the <<team>> team on <<weekday>> as a <<job>>.\n\nCheers",
        "Subject: Expense policy\n\nA reminder that receipts over $75 must be attached to expense reports within 30 days.\n\nFinance",
    ], filler=True),
    F("email/shipping", "email", "neg", [
        "Subject: Your order has shipped\n\nOrder <<order_ref>> is on its way. Tracking: <<tracking>>. Estimated delivery <<date>>.\n\n<<company>>",
        "Subject: Invoice <<invoice_no>>\n\nPlease find invoice <<invoice_no>> for <<amount>> attached. Payment terms: net 30.\n\n<<work_alias>>",
        "Subject: Build report\n\nCommit <<gitsha>> passed. Artifact checksum <<sha>>, run id <<uuid>>.\n\nCI",
    ], hard_case="look-alike-number"),
    F("email/masked", "email", "neg", [
        "Subject: Payment received\n\nWe received <<amount>> from the card ending in <<card_tail>>. Thank you!\n\n<<company>>",
        "Subject: Account verification\n\nThe SSN on your application (<<masked_ssn>>) matches our records. No further action is needed.\n\nCustomer Care",
        "Subject: Payout sent\n\nYour payout was sent to account <<masked_account>>.\n\n<<work_alias>>",
    ], hard_case="masked"),
    F("email/doc-values", "email", "neg", [
        "Subject: QA test plan\n\nFor checkout tests use card <<doc_card>>, any CVC and a future expiry. Never use real cards in QA.\n\n<<first_name>>",
        "Subject: SDK setup\n\nIn the quickstart, set aws_access_key_id = <<doc_aws_key_id>> and aws_secret_access_key = <<doc_aws_secret>> as placeholders, then replace them with your own.\n\nDocs team",
        "Subject: Config template\n\nThe template uses api_key: YOUR_API_KEY and password: <password>. Fill them in locally only.\n\n<<first_name>>",
    ], hard_case="documentation-value"),
    F("email/topic-only", "email", "neg", [
        "Subject: Data handling reminder\n\nSSNs and card numbers must never be stored in spreadsheets. Use the approved vault and report any exposure to security.\n\nSecurity team",
        "Subject: Key rotation\n\nWe are moving to 90-day API key rotation. Owners will get a reminder two weeks before each key expires.\n\nPlatform team",
        "Subject: Privacy training\n\nThis quarter's training covers HIPAA, PCI DSS and how to handle a customer who emails a password.\n\nCompliance",
    ], hard_case="topic-only"),
    F("email/public-business", "email", "neg", [
        "Subject: Press release\n\n<<company>>, headquartered at <<office_address>>, today announced <<exec_name>> as <<exec_title>>. <<exec_name>> joins from <<company2>>.\n\nMedia relations",
        "Subject: Partner announcement\n\n<<company>> (NASDAQ: <<ticker>>) and <<company2>> announced a partnership today. Read the full release on our website.\n\nCommunications",
        "Subject: Event recap\n\n<<exec_name>>, <<exec_title>> of <<company>>, spoke at the summit in <<city>> about the company's published strategy.\n\nMarketing",
    ], hard_case="public-business"),
    F("email/first-name-only", "email", "neg", [
        "Subject: Thanks\n\nThanks <<first_name>> for covering on-call this weekend!\n\n<<first_name2>>",
        "Subject: Intro\n\n<<first_name2>>, meet <<first_name>>, who leads the <<team>> work. I'll let you two connect.\n\n<<first_name3>>",
        "Subject: Room booking\n\n<<first_name>> booked the large room for <<weekday>> at <<time>>.\n\nFacilities",
    ], hard_case="first-name-only"),
    F("email/number-no-keyword", "email", "neg", [
        "Subject: Case update\n\nCase <<ninedigit>> was escalated to tier 2.\n\nSupport",
        "Subject: Parts\n\nPart number <<ninedigit>> is discontinued; the replacement is <<ninedigit2>>.\n\nProcurement",
        "Subject: Reservation\n\nYour reservation <<ninedigit>> is confirmed for <<date>>.\n\n<<company>>",
    ], hard_case="number-without-keyword"),

    # ======================================================================= GenAI prompts (W3)
    F("genai/rewrite-personal", "genai", "pos", [
        "Rewrite my CV header to look more professional: <<full_name:PERSONAL>>, born <<dob:PERSONAL>>, <<home_address:PERSONAL>>, <<personal_email:PERSONAL>>",
        "Write a short bio for the school newsletter. Details: <<full_name:PERSONAL>>, lives at <<home_address:PERSONAL>>, cell <<phone:PERSONAL>>.",
        "Draft a cover letter. My info: <<full_name:PERSONAL>> | <<personal_email:PERSONAL>> | <<phone:PERSONAL>>",
    ], W3, category="PERSONAL"),
    F("genai/form-fill", "genai", "pos", [
        "Fill in this visa application for me. Name: <<full_name>>, passport number <<passport:GOV_ID>>, SSN <<ssn:GOV_ID>>.",
        "Check my tax return numbers. SSN <<ssn:GOV_ID>>, wages <<amount>>, filing single.",
        "Translate to Spanish: 'My driver's license number is <<drivers_license:GOV_ID>> and it expires on <<date>>.'",
    ], W3, category="GOV_ID"),
    F("genai/finance", "genai", "pos", [
        "Draft an invoice from <<company>> for <<amount>>. Pay to IBAN <<iban:FINANCIAL>>, account holder <<full_name>>.",
        "Write a polite email asking my landlord to charge rent to my card <<card:FINANCIAL>> (exp <<expiry>>).",
        "Summarize my bank details in a table: routing <<routing:FINANCIAL>>, account <<bank_account:FINANCIAL>>, bank <<company>>.",
    ], W3, category="FINANCIAL"),
    F("genai/health", "genai", "pos", [
        "Write an appeal letter to my insurer. Patient: <<full_name:HEALTH>>, member ID <<insurance_id:HEALTH>>, denied claim for <<procedure:HEALTH>>.",
        "Explain these results in simple words for <<full_name:HEALTH>>: diagnosis <<diagnosis:HEALTH>>, prescribed <<medication:HEALTH>>.",
        "Summarize this note: <<full_name:HEALTH>>, MRN <<mrn:HEALTH>>, admitted <<date>> for <<diagnosis:HEALTH>>.",
    ], W3, category="HEALTH"),
    F("genai/confidential", "genai", "pos", [
        "Summarize this memo in three bullets: '<<conf_marker:CONFIDENTIAL>>. <<deal:CONFIDENTIAL>>.'",
        "Turn this into a slide: <<conf_marker:CONFIDENTIAL>> - <<unreleased:CONFIDENTIAL>>.",
        "Make this roadmap sound exciting for an all-hands: '<<conf_marker:CONFIDENTIAL>>: <<roadmap:CONFIDENTIAL>>'",
        "Format our customer list as a table (<<conf_marker:CONFIDENTIAL>>): <<customer_list:CONFIDENTIAL>>",
    ], W3, category="CONFIDENTIAL"),
    F("genai/credentials", "genai", "pos", [
        "My CI job fails with 401. Here's the env: GITHUB_TOKEN=<<github_pat:CREDENTIALS>> REPO=<<repo>>. What's wrong?",
        "Why does this curl return 403? curl -H 'Authorization: Bearer <<jwt:CREDENTIALS>>' https://api.<<domain>>/v1/me",
        "Convert this to a Kubernetes secret: STRIPE_SECRET_KEY=<<stripe_key:CREDENTIALS>>",
        "Explain this error. I'm using npm token <<npm_token:CREDENTIALS>> and publish fails with E403.",
    ], W3, category="CREDENTIALS"),
    F("genai/benign", "genai", "neg", [
        "Write a haiku about autumn in <<city>>.",
        "Explain the difference between TCP and UDP in two paragraphs.",
        "Give me five ideas for a team-building activity for 12 people.",
        "Summarize the plot of Pride and Prejudice in 100 words.",
        "What is a good name for a coffee shop next to a library?",
        "Translate 'the meeting moved to <<weekday>> at <<time>>' into French.",
        "Draft a polite reminder that timesheets are due on <<date>>.",
        "Explain what a p99 latency is to a non-technical manager.",
    ], W3),
    F("genai/topic-only", "genai", "neg", [
        "How should a small company store customer SSNs to comply with US law?",
        "What is the safest way to rotate AWS access keys without downtime?",
        "Explain PCI DSS scope reduction with tokenization.",
        "Write a policy paragraph that forbids sharing passwords over chat.",
    ], W3, hard_case="topic-only"),
    F("genai/doc-values", "genai", "neg", [
        "Why does boto3 reject the example key <<doc_aws_key_id>>? I copied it from the docs.",
        "Write a unit test that uses Stripe's test card 4242 4242 4242 4242.",
        "Fill in a sample .env with API_KEY=YOUR_API_KEY and DB_PASSWORD=<password> placeholders.",
    ], W3, hard_case="documentation-value"),
    F("genai/look-alike", "genai", "neg", [
        "Why would order <<order_ref>> show up twice in our export?",
        "Is <<uuid>> a valid UUID v4? Explain the format.",
        "Parse this tracking number and tell me the carrier: <<tracking>>",
    ], W3, hard_case="look-alike-number"),
    F("genai/public-business", "genai", "neg", [
        "Write a LinkedIn post congratulating <<exec_name>> on becoming <<exec_title>> of <<company>>.",
        "Summarize this press release: '<<company>> opens a new office at <<office_address>>.'",
        "Compare <<company>> (<<ticker>>) and <<company2>> based on their public annual reports.",
    ], W3, hard_case="public-business"),
    F("genai/masked", "genai", "neg", [
        "Explain why a receipt shows only <<masked_card>> instead of the whole number.",
        "Our form shows the SSN as <<masked_ssn>>. Is that compliant display masking?",
        "Write a support reply: refund went to the card ending <<card_tail>>, allow 5 days.",
    ], W3, hard_case="masked"),

    # ======================================================================= code questions (W3)
    F("code_question/aws-keys", "code_question", "pos", [
        "Why does this give AccessDenied?\n```python\nimport boto3\ns3 = boto3.client('s3', aws_access_key_id='<<aws_key_id:CREDENTIALS>>',\n                  aws_secret_access_key='<<aws_secret:CREDENTIALS>>')\nprint(s3.list_buckets())\n```",
        "My Terraform plan fails:\n```hcl\nprovider \"aws\" {\n  region     = \"<<region>>\"\n  access_key = \"<<aws_key_id:CREDENTIALS>>\"\n  secret_key = \"<<aws_secret:CREDENTIALS>>\"\n}\n```",
        "This works locally but not in Lambda, why?\n```js\nconst client = new S3Client({ credentials: { accessKeyId: '<<aws_key_id:CREDENTIALS>>', secretAccessKey: '<<aws_secret:CREDENTIALS>>' } });\n```",
    ], W3, category="CREDENTIALS"),
    F("code_question/db-uri", "code_question", "pos", [
        "SQLAlchemy can't connect:\n```python\nengine = create_engine('<<db_uri:CREDENTIALS>>')\n```\nWhat am I missing?",
        "Why is my Go service timing out?\n```go\ndb, err := sql.Open(\"postgres\", \"<<db_uri:CREDENTIALS>>\")\n```",
        "Help me fix this docker-compose:\n```yaml\nservices:\n  api:\n    environment:\n      DATABASE_URL: <<db_uri:CREDENTIALS>>\n```",
    ], W3, category="CREDENTIALS"),
    F("code_question/fixture-values", "code_question", "pos", [
        "This test fails on the date parsing, why?\n```python\nuser = {'name': '<<full_name:PERSONAL>>', 'dob': '<<dob:PERSONAL>>', 'email': '<<personal_email:PERSONAL>>'}\nassert parse_user(user).age > 18\n```",
        "Why does the regex not match the card?\n```js\nconst card = '<<card:FINANCIAL>>';\nconsole.log(/^\\d{16}$/.test(card));\n```",
        "Validate this SSN format in Java:\n```java\nString ssn = \"<<ssn:GOV_ID>>\";\nboolean ok = ssn.matches(\"\\\\d{3}-\\\\d{2}-\\\\d{4}\");\n```",
        "Why does Luhn validation fail here?\n```python\nassert luhn_valid('<<card:FINANCIAL>>')\n```",
        "This passport regex rejects a valid number:\n```python\nre.fullmatch(r'[A-Z]\\d{8}', '<<passport:GOV_ID>>')\n```",
        "Why is age negative?\n```python\nprofile = {'name': '<<full_name:PERSONAL>>', 'born': '<<dob:PERSONAL>>', 'home': '<<home_address:PERSONAL>>'}\n```",
    ], W3),
    F("code_question/benign", "code_question", "neg", [
        "Why is this list comprehension slow?\n```python\nresult = [x * 2 for x in range(10_000_000) if x % 3 == 0]\n```",
        "How do I debounce this handler?\n```js\nwindow.addEventListener('resize', () => recalcLayout());\n```",
        "Why does this goroutine leak?\n```go\nfor _, j := range jobs {\n    go func() { process(j) }()\n}\n```",
        "Fix the off-by-one:\n```java\nfor (int i = 0; i <= items.size(); i++) { total += items.get(i); }\n```",
        "Terraform says the bucket already exists:\n```hcl\nresource \"aws_s3_bucket\" \"logs\" {\n  bucket = \"<<service>>-logs-<<region>>\"\n}\n```",
    ], W3),
    F("code_question/doc-values", "code_question", "neg", [
        "Is this the right way to mock credentials in tests?\n```python\nos.environ['AWS_ACCESS_KEY_ID'] = '<<doc_aws_key_id>>'\nos.environ['AWS_SECRET_ACCESS_KEY'] = '<<doc_aws_secret>>'\n```",
        "The Stripe test fails:\n```js\nawait pay({ number: '4242424242424242', exp: '12/34', cvc: '123' });\n```",
        "Why does the app ignore my key?\n```yaml\napi_key: YOUR_API_KEY\n```",
    ], W3, hard_case="documentation-value"),
    F("code_question/indirect", "code_question", "neg", [
        "Is reading secrets like this OK?\n```python\npassword = os.environ['DB_PASSWORD']\nconn = psycopg.connect(host=HOST, password=password)\n```",
        "How do I reference a secret in Terraform?\n```hcl\npassword = data.aws_secretsmanager_secret_version.db.secret_string\n```",
        "Is this Helm value safe?\n```yaml\ndb:\n  passwordSecretRef: <<service>>-db-credentials\n```",
    ], W3, hard_case="indirect-reference"),
    F("code_question/masked", "code_question", "neg", [
        "How do I format the masked value like this?\n```python\nprint(mask(card))  # -> <<masked_card>>\n```",
        "Unit test for SSN masking:\n```python\nassert mask_ssn(value) == '<<masked_ssn>>'\n```",
        "Why does the log show the card like this?\n```text\npayment ok card=<<masked_card>>\n```",
    ], W3, hard_case="masked"),

    # ======================================================================= memos (W1)
    F("memo/confidential", "memo", "pos", [
        "<<conf_marker:CONFIDENTIAL>>\nBoard update, <<date>>\n\n<<deal:CONFIDENTIAL>>. Integration planning starts next week; only the deal team is informed.",
        "<<conf_marker:CONFIDENTIAL>>\nTo: Executive staff\nRe: <<quarter>> results\n\n<<unreleased:CONFIDENTIAL>>. The earnings script is being drafted by finance.",
        "<<conf_marker:CONFIDENTIAL>>\nProduct strategy\n\n<<roadmap:CONFIDENTIAL>>. Sales should not mention this to customers yet.",
        "<<conf_marker:CONFIDENTIAL>>\nKey accounts at risk\n\n<<customer_list:CONFIDENTIAL>>. Account managers should schedule reviews this month.",
    ], W1, category="CONFIDENTIAL"),
    F("memo/benign", "memo", "neg", [
        "Facilities update, <<date>>\n\nThe third floor kitchen reopens on <<weekday>>. Please label food in the shared fridge; it is cleared every Friday.",
        "Team update\n\nThe <<team>> team shipped <<product>> <<version>> this sprint. Next sprint focuses on documentation and bug fixes.",
        "Holiday schedule\n\nThe office is closed on <<date>> and <<date2>>. Support keeps normal on-call coverage.",
        "IT notice\n\nLaptops will receive the <<version>> update on <<weekday>>. Save your work before <<time>>; restarts take about ten minutes.",
        "Sustainability\n\nWe recycled <<percent>> of office waste last quarter. New compost bins arrive on <<date>>.",
        "Training\n\nA new course on writing clear design docs is available on the learning portal. It takes about an hour.",
        "Customer support\n\nResponse times improved to <<num>> minutes on average. Thanks to everyone who joined the weekend rotation.",
    ], W1, filler=True),
    F("memo/public-business", "memo", "neg", [
        "PRESS RELEASE\n\n<<company>>, headquartered at <<office_address>>, today announced that <<exec_name>> will join as <<exec_title>>. <<exec_name>> previously led operations at <<company2>>.",
        "Published annual report summary\n\n<<company>> (<<ticker>>) reported revenue growth of <<percent>> in its published annual report. The CEO, <<exec_name>>, presented the results at the public shareholder meeting in <<city>>.",
        "Public announcement\n\n<<company>> opened a new office at <<office_address>>. The opening was attended by <<exec_name>>, <<exec_title>>.",
    ], W1, hard_case="public-business"),
    F("memo/topic-only", "memo", "neg", [
        "Data handling standard\n\nSocial security numbers, card numbers and health data must be stored only in approved systems with encryption at rest. Report any exposure to the security team within 24 hours.",
        "Secrets management\n\nAPI keys and passwords are issued through the vault. Keys rotate every 90 days. Never send a secret by email or chat.",
        "Privacy notice update\n\nWe updated how we describe the categories of personal data we process. The change does not affect any customer settings.",
    ], W1, hard_case="topic-only"),

    # ======================================================================= letters and forms (W1)
    F("letter/hr", "letter", "pos", [
        "Employment verification\n\nThis letter confirms that <<full_name>> (SSN <<ssn:GOV_ID>>) has been employed as a <<job>> since <<date>>.\n\nHuman Resources, <<company>>",
        "Offer details\n\nDear <<first_name>>,\n\nTo complete your file we recorded your passport number <<passport:GOV_ID>> and tax ID <<itin:GOV_ID>>.\n\nPeople Operations",
        "Benefits enrollment\n\nEmployee: <<full_name>>\nSSN: <<ssn:GOV_ID>>\nPlan: PPO family\nStart: <<date>>",
        "Right to work check\n\nWe copied the driver's license <<drivers_license:GOV_ID>> presented by <<full_name>> on <<date>>.\n\nHR",
    ], W1, category="GOV_ID"),
    F("letter/medical", "letter", "pos", [
        "Discharge summary\n\nPatient: <<full_name:HEALTH>>\nMRN: <<mrn:HEALTH>>\nDiagnosis: <<diagnosis:HEALTH>>\nPlan: <<medication:HEALTH>>, follow-up in four weeks.",
        "To whom it may concern,\n\n<<full_name:HEALTH>> is under my care for <<diagnosis:HEALTH>> and should avoid heavy lifting until <<date>>.\n\nDr. <<last_name2>>",
        "Prior authorization request\n\nMember: <<full_name:HEALTH>>, ID <<insurance_id:HEALTH>>, blood type <<blood_type:HEALTH>>. Requested: <<procedure:HEALTH>>.",
    ], W1, category="HEALTH"),
    F("letter/customer", "letter", "pos", [
        "Account opening confirmation\n\nDear <<full_name:PERSONAL>>,\n\nWe have recorded your date of birth <<dob:PERSONAL>> and home address <<home_address:PERSONAL>>.\n\n<<company>> Customer Service",
        "Delivery note\n\nDeliver to: <<full_name:PERSONAL>>, <<home_address:PERSONAL>>, phone <<phone:PERSONAL>>.",
        "Membership card\n\nName: <<full_name:PERSONAL>>\nBorn: <<dob:PERSONAL>>\nEmail: <<personal_email:PERSONAL>>",
    ], W1, category="PERSONAL"),
    F("letter/statement", "letter", "pos", [
        "Card statement\n\nCard number: <<card:FINANCIAL>>\nStatement date: <<date>>\nBalance due: <<amount>>",
        "Wire confirmation\n\nBeneficiary: <<full_name>>\nIBAN: <<iban:FINANCIAL>>\nAmount: <<amount>>",
        "Direct debit mandate\n\nAccount holder: <<full_name>>\nRouting: <<routing:FINANCIAL>>\nAccount: <<bank_account:FINANCIAL>>",
        "Payroll confirmation\n\n<<full_name>>, your salary will be paid to account <<bank_account:FINANCIAL>> at routing <<routing:FINANCIAL>> from <<date>>.\n\nPayroll",
    ], W1, category="FINANCIAL"),
    F("letter/benign", "letter", "neg", [
        "Dear customer,\n\nThank you for contacting <<company>>. Your request <<ticket>> was resolved. We value your feedback.\n\nCustomer Service",
        "Notice of price change\n\nFrom <<date>>, the monthly price of the <<product>> plan changes to <<amount>>. No action is needed.\n\n<<company>>",
        "Event invitation\n\nYou are invited to our open house on <<date>> at <<office_address>>. Light refreshments will be served.\n\n<<company>>",
        "Service update\n\nOur support hours change to 8:00-20:00 from <<date>>. Thank you for being a customer.\n\n<<company>>",
        "Hello <<first_name>>,\n\nYour library books are due on <<date>>. You can renew them online.\n\nCity Library",
    ], W1, filler=True),
    F("letter/masked", "letter", "neg", [
        "Account summary\n\nName on account: <<full_name>>\nSSN: <<masked_ssn>>\nCard: <<masked_card>>\nDate of birth: <<masked_dob>>",
        "Payment receipt\n\nPaid by card ending in <<card_tail>>. Account <<masked_account>>. Thank you, <<first_name>>.",
        "Verification letter\n\nWe verified the identity of <<full_name>> using a document ending in <<card_tail>>. Phone on file: <<masked_phone>>.",
    ], W1, hard_case="masked"),
    F("letter/number-no-keyword", "letter", "neg", [
        "Shipment notice\n\nReference <<ninedigit>>, <<num>> cartons, delivered to the dock on <<date>>.",
        "Warranty registration\n\nSerial <<ninedigit>> registered on <<date>>. Coverage: two years.",
        "Membership renewal\n\nMember number <<ninedigit>> renews on <<date>>. No action is needed.",
    ], W1, hard_case="number-without-keyword"),
    F("letter/first-name-only", "letter", "neg", [
        "Hi <<first_name>>,\n\nThanks for visiting our store. Your loyalty points balance is <<big_num>>.\n\n<<company>>",
        "Dear <<first_name>>,\n\nYour appointment with the help desk is confirmed for <<weekday>> at <<time>>.\n\nIT",
        "Hello <<first_name>>,\n\nYour feedback about the <<product>> app reached the product team. Thank you!\n\n<<company>>",
    ], W1, hard_case="first-name-only"),

    # ======================================================================= logs (W1)
    F("log/benign", "log", "neg", [
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "<<service>>", "path": "<<path>>", "status": <<status>>, "latency_ms": <<ms>>, "request_id": "<<uuid>>"}',
        '<<ip>> - - [<<iso_ts>>] "GET <<path>> HTTP/1.1" <<status>> <<big_num>> "-" "Mozilla/5.0"',
        '{"ts": "<<iso_ts>>", "level": "WARN", "service": "<<service>>", "msg": "retrying upstream call", "attempt": <<num>>, "trace_id": "<<uuid>>"}',
        '<<iso_ts>> INFO <<service>> job=<<service2>>-sync rows=<<big_num>> duration_ms=<<ms>>',
        '{"ts": "<<iso_ts>>", "level": "DEBUG", "service": "<<service>>", "msg": "cache hit", "key": "<<service2>>:<<uuid>>"}',
        '<<iso_ts>> ERROR <<service>> timeout calling <<host>>:<<port>> after <<ms>>ms',
    ], W1, filler=True),
    F("log/sensitive", "log", "pos", [
        '{"ts": "<<iso_ts>>", "level": "DEBUG", "service": "checkout", "msg": "charge request", "card": "<<card:FINANCIAL>>", "cvv": "<<cvv:FINANCIAL>>", "amount": "<<amount>>"}',
        '<<iso_ts>> DEBUG payments request body={"pan":"<<card:FINANCIAL>>","exp":"<<expiry>>"}',
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "kyc", "msg": "verification submitted", "ssn": "<<ssn:GOV_ID>>", "request_id": "<<uuid>>"}',
        '<<ip>> - - [<<iso_ts>>] "GET /api/v1/me HTTP/1.1" 200 512 "-" "curl/8.4" Authorization="Bearer <<jwt:CREDENTIALS>>"',
        '<<iso_ts>> WARN auth login failed user=<<username>> password=<<password:CREDENTIALS>> ip=<<ip>>',
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "profile", "event": "update", "name": "<<full_name:PERSONAL>>", "dob": "<<dob:PERSONAL>>", "email": "<<personal_email:PERSONAL>>"}',
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "ehr", "event": "record_viewed", "patient": "<<full_name:HEALTH>>", "mrn": "<<mrn:HEALTH>>", "dx": "<<diagnosis:HEALTH>>"}',
        '<<iso_ts>> DEBUG http set-cookie: <<session_cookie:CREDENTIALS>>; HttpOnly; Secure',
    ], W1),
    F("log/look-alike", "log", "neg", [
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "orders", "msg": "order shipped", "order_id": "<<order_ref>>", "tracking": "<<tracking>>"}',
        '<<iso_ts>> INFO artifacts uploaded sha256=<<sha>> commit=<<gitsha>>',
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "billing", "msg": "invoice created", "invoice": "<<invoice_no>>", "ref": "<<order_ref>>"}',
    ], W1, hard_case="look-alike-number"),
    F("log/masked", "log", "neg", [
        '{"ts": "<<iso_ts>>", "level": "INFO", "service": "checkout", "msg": "charge ok", "card": "<<masked_card>>", "amount": "<<amount>>"}',
        '<<iso_ts>> INFO kyc verified ssn=<<masked_ssn>> request_id=<<uuid>>',
        '<<iso_ts>> WARN auth login failed user=<<username>> password=<<masked_secret>> ip=<<ip>>',
    ], W1, hard_case="masked"),

    # ======================================================================= config (W1)
    F("config/benign", "config", "neg", [
        "# <<service>> settings\nLOG_LEVEL=info\nPORT=<<port>>\nREGION=<<region>>\nFEATURE_NEW_CHECKOUT=true\nCACHE_TTL_SECONDS=<<num>>0",
        "server:\n  port: <<port>>\n  threads: <<num>>\nlogging:\n  level: info\n  format: json\nfeatures:\n  dark_mode: true\n  beta_search: false",
        '{\n  "service": "<<service>>",\n  "region": "<<region>>",\n  "retries": <<num>>,\n  "timeout_ms": <<ms>>,\n  "upstream": "https://<<host>>"\n}',
        "[database]\nhost = <<host>>\nport = 5432\npool_size = <<num>>\n\n[metrics]\nenabled = true\ninterval = 60",
        "replicas: <<num>>\nimage: <<service>>:<<version>>\nresources:\n  cpu: 500m\n  memory: 512Mi\nenv: <<env_name>>",
    ], W1, filler=True),
    F("config/secrets", "config", "pos", [
        "# <<service>> production\nDATABASE_URL=<<db_uri:CREDENTIALS>>\nSTRIPE_SECRET_KEY=<<stripe_key:CREDENTIALS>>",
        "aws:\n  access_key_id: <<aws_key_id:CREDENTIALS>>\n  secret_access_key: <<aws_secret:CREDENTIALS>>\n  region: <<region>>",
        '{\n  "slack_bot_token": "<<slack_token:CREDENTIALS>>",\n  "channel": "#<<team>>-alerts"\n}',
        "GITHUB_TOKEN=<<github_pat:CREDENTIALS>>\nNPM_TOKEN=<<npm_token:CREDENTIALS>>\nCI=true",
        "smtp:\n  host: smtp.<<domain>>\n  user: <<work_alias>>\n  password: <<password:CREDENTIALS>>",
        "# service account key\nGOOGLE_API_KEY=<<google_api_key:CREDENTIALS>>\nOPENAI_API_KEY=<<openai_key:CREDENTIALS>>",
    ], W1, category="CREDENTIALS"),
    F("config/test-labeled-secret", "config", "pos", [
        "# test credentials, delete before release\nDB_PASSWORD=<<password:CREDENTIALS>>",
        "# sandbox only - not a real key\nSTRIPE_SECRET_KEY=<<stripe_key:CREDENTIALS>>",
        "test:\n  # dummy values for local runs\n  aws_access_key_id: <<aws_key_id:CREDENTIALS>>\n  aws_secret_access_key: <<aws_secret:CREDENTIALS>>",
    ], W1, category="CREDENTIALS", hard_case="test-labeled-secret"),
    F("config/doc-values", "config", "neg", [
        "# .env.example - copy to .env and fill in\nAPI_KEY=YOUR_API_KEY\nDB_PASSWORD=<password>",
        "# values from the AWS documentation\naws_access_key_id = <<doc_aws_key_id>>\naws_secret_access_key = <<doc_aws_secret>>",
        "payments:\n  test_card: \"<<doc_card>>\"\n  mode: sandbox",
    ], W1, hard_case="documentation-value"),
    F("config/indirect", "config", "neg", [
        "DATABASE_PASSWORD=${DB_PASSWORD}\nSTRIPE_KEY_PATH=vault:secret/data/<<service>>/stripe",
        "db:\n  host: <<host>>\n  password_from: secretsmanager://<<service>>/db\n  user: app",
        '{\n  "api_key_env": "PARTNER_API_KEY",\n  "token_file": "/run/secrets/<<service>>-token"\n}',
    ], W1, hard_case="indirect-reference"),
]


def _code(lang: str, benign: list[str], secret: list[str], conf: list[str], pii: list[str] | None = None) -> list[Family]:
    ch = f"code-{lang}"
    fams = [F(f"{ch}/benign", ch, "neg", benign, W1, filler=True),
            F(f"{ch}/hardcoded-secret", ch, "pos", secret, W1, category="CREDENTIALS"),
            F(f"{ch}/internal-only", ch, "pos", conf, W1, category="CONFIDENTIAL")]
    if pii:
        fams.append(F(f"{ch}/fixture-pii", ch, "pos", pii, W1, category="PERSONAL"))
    return fams


FAMILIES += _code("py", benign=[
    "def chunk(items, size):\n    \"\"\"Yield successive chunks of a list.\"\"\"\n    for i in range(0, len(items), size):\n        yield items[i:i + size]",
    "class RetryPolicy:\n    def __init__(self, attempts=<<num>>, backoff=0.5):\n        self.attempts = attempts\n        self.backoff = backoff\n\n    def delay(self, n):\n        return self.backoff * 2 ** n",
    "def parse_duration(text: str) -> int:\n    units = {'s': 1, 'm': 60, 'h': 3600}\n    return int(text[:-1]) * units[text[-1]]",
    "@app.get('/health')\ndef health():\n    return {'status': 'ok', 'version': '<<version>>'}",
    "def median(values):\n    s = sorted(values)\n    mid = len(s) // 2\n    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2",
    "import logging\n\nlog = logging.getLogger('<<service>>')\n\ndef handle(event):\n    log.info('received %d records', len(event['records']))\n    return [r['id'] for r in event['records']]",
], secret=[
    "STRIPE_KEY = '<<stripe_key:CREDENTIALS>>'\n\ndef charge(amount):\n    return stripe.Charge.create(amount=amount, api_key=STRIPE_KEY)",
    "client = boto3.client(\n    's3',\n    aws_access_key_id='<<aws_key_id:CREDENTIALS>>',\n    aws_secret_access_key='<<aws_secret:CREDENTIALS>>',\n)",
    "SLACK_TOKEN = \"<<slack_token:CREDENTIALS>>\"\n\ndef notify(text):\n    requests.post('https://slack.com/api/chat.postMessage', headers={'Authorization': f'Bearer {SLACK_TOKEN}'}, json={'text': text})",
    "engine = create_engine('<<db_uri:CREDENTIALS>>')\nSession = sessionmaker(bind=engine)",
], conf=[
    "# Copyright <<company>>. <<conf_marker:CONFIDENTIAL>>. Proprietary pricing logic.\ndef enterprise_discount(arr):\n    return 0.32 if arr > 500_000 else 0.18",
    "# <<conf_marker:CONFIDENTIAL>> - fraud scoring rules, do not publish\nTHRESHOLDS = {'velocity': 7, 'geo_mismatch': 0.83, 'device_age_days': 3}",
    "\"\"\"<<conf_marker:CONFIDENTIAL>>: churn model features for <<product>>.\"\"\"\nFEATURES = ['seats_drop_30d', 'support_tickets_90d', 'exec_sponsor_left']",
], pii=[
    "TEST_USER = {\n    'name': '<<full_name:PERSONAL>>',\n    'dob': '<<dob:PERSONAL>>',\n    'address': '<<home_address:PERSONAL>>',\n}",
    "def test_export():\n    row = export_customer('<<full_name:PERSONAL>>', '<<personal_email:PERSONAL>>', '<<phone:PERSONAL>>')\n    assert row['ok']",
    "SEED_CUSTOMERS = [('<<full_name:PERSONAL>>', '<<personal_email:PERSONAL>>', '<<dob:PERSONAL>>')]",
])
FAMILIES += _code("js", benign=[
    "export function debounce(fn, ms) {\n  let t;\n  return (...args) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), ms);\n  };\n}",
    "app.get('/api/items', async (req, res) => {\n  const items = await db.items.findMany({ take: <<num>> });\n  res.json(items);\n});",
    "export const formatPrice = (cents) => `$${(cents / 100).toFixed(2)}`;",
    "function groupBy(list, key) {\n  return list.reduce((acc, x) => {\n    (acc[x[key]] ||= []).push(x);\n    return acc;\n  }, {});\n}",
    "const config = {\n  region: '<<region>>',\n  retries: <<num>>,\n  timeoutMs: <<ms>>,\n};\nexport default config;",
], secret=[
    "const stripe = require('stripe')('<<stripe_key:CREDENTIALS>>');",
    "const octokit = new Octokit({ auth: '<<github_pat:CREDENTIALS>>' });",
    "const openai = new OpenAI({ apiKey: '<<openai_key:CREDENTIALS>>' });",
    "const pool = new Pool({ connectionString: '<<db_uri:CREDENTIALS>>' });",
], conf=[
    "// <<conf_marker:CONFIDENTIAL>> - <<company>> partner pricing\nexport const PARTNER_MARGIN = { gold: 0.27, silver: 0.19 };",
    "/* <<conf_marker:CONFIDENTIAL>>: unreleased <<product>> 3.0 feature flags */\nexport const FLAGS = { agentMode: true, newBilling: true };",
    "// <<conf_marker:CONFIDENTIAL>>. Ranking weights for the <<product>> feed.\nexport const WEIGHTS = { recency: 0.41, affinity: 0.37, paid: 0.22 };",
])
FAMILIES += _code("go", benign=[
    "func Max(a, b int) int {\n\tif a > b {\n\t\treturn a\n\t}\n\treturn b\n}",
    "func (s *Server) Health(w http.ResponseWriter, r *http.Request) {\n\tw.WriteHeader(http.StatusOK)\n\tw.Write([]byte(\"ok\"))\n}",
    "type Config struct {\n\tRegion  string\n\tRetries int\n\tTimeout time.Duration\n}",
    "func Sum(xs []float64) (t float64) {\n\tfor _, x := range xs {\n\t\tt += x\n\t}\n\treturn\n}",
], secret=[
    "const apiKey = \"<<google_api_key:CREDENTIALS>>\"\n\nfunc client() *maps.Client {\n\tc, _ := maps.NewClient(maps.WithAPIKey(apiKey))\n\treturn c\n}",
    "db, err := sql.Open(\"postgres\", \"<<db_uri:CREDENTIALS>>\")\nif err != nil {\n\tlog.Fatal(err)\n}",
    "var token = \"<<slack_token:CREDENTIALS>>\" // bot token",
], conf=[
    "// <<conf_marker:CONFIDENTIAL>>. Bid strategy for <<product>> ads.\nconst maxBidMultiplier = 3.7",
    "// <<conf_marker:CONFIDENTIAL>>: settlement netting rules, <<company>> treasury\nconst nettingWindow = 36 * time.Hour",
])
FAMILIES += _code("java", benign=[
    "public static int clamp(int v, int lo, int hi) {\n    return Math.max(lo, Math.min(hi, v));\n}",
    "@GetMapping(\"/health\")\npublic Map<String, String> health() {\n    return Map.of(\"status\", \"ok\");\n}",
    "public record Order(String id, int quantity, BigDecimal price) {}",
    "public List<String> names(List<User> users) {\n    return users.stream().map(User::name).sorted().toList();\n}",
], secret=[
    "private static final String DB_PASSWORD = \"<<password:CREDENTIALS>>\";",
    "AWSCredentials creds = new BasicAWSCredentials(\"<<aws_key_id:CREDENTIALS>>\", \"<<aws_secret:CREDENTIALS>>\");",
    "String jwt = \"<<jwt:CREDENTIALS>>\"; // service token",
], conf=[
    "/* <<conf_marker:CONFIDENTIAL>>. <<company>> underwriting rules. */\npublic static final double MAX_DTI = 0.43;",
    "// <<conf_marker:CONFIDENTIAL>>: unreleased loyalty tiers\npublic enum Tier { BRONZE, SILVER, GOLD, OBSIDIAN }",
])
FAMILIES += _code("tf", benign=[
    "resource \"aws_s3_bucket\" \"logs\" {\n  bucket = \"<<service>>-logs-<<region>>\"\n  tags = {\n    team = \"<<team>>\"\n  }\n}",
    "variable \"instance_count\" {\n  type    = number\n  default = <<num>>\n}",
    "module \"vpc\" {\n  source = \"terraform-aws-modules/vpc/aws\"\n  cidr   = \"10.0.0.0/16\"\n  azs    = [\"<<region>>a\", \"<<region>>b\"]\n}",
    "output \"bucket_arn\" {\n  value = aws_s3_bucket.logs.arn\n}",
], secret=[
    "provider \"aws\" {\n  region     = \"<<region>>\"\n  access_key = \"<<aws_key_id:CREDENTIALS>>\"\n  secret_key = \"<<aws_secret:CREDENTIALS>>\"\n}",
    "resource \"aws_db_instance\" \"main\" {\n  engine   = \"postgres\"\n  username = \"app\"\n  password = \"<<password:CREDENTIALS>>\"\n}",
    "provider \"github\" {\n  token = \"<<github_pat:CREDENTIALS>>\"\n}",
], conf=[
    "# <<conf_marker:CONFIDENTIAL>>: <<codename>> launch infrastructure (unannounced)\nresource \"aws_ecs_service\" \"launch\" {\n  name = \"<<codename>>-api\"\n}",
    "# <<conf_marker:CONFIDENTIAL>> - acquisition data room\nresource \"aws_s3_bucket\" \"dataroom\" {\n  bucket = \"<<codename>>-dataroom\"\n}",
])

CODE_CHANNELS = ("code-py", "code-js", "code-go", "code-java", "code-tf")
W1_S3_CHANNELS = ("email", "memo", "letter", "log", "config", *CODE_CHANNELS)
W3_CHANNELS = ("chat", "email", "genai", "code_question")


def by_name() -> dict[str, Family]:
    return {f.name: f for f in FAMILIES}


def validate() -> None:
    """Static checks on the template families (run by the build and the tests)."""
    from tools.dspm.generator import (
        CATEGORIES,
        PLACEHOLDER,
        PROVIDERS,
        template_categories,
        templates_for,
    )

    names = [f.name for f in FAMILIES]
    if len(names) != len(set(names)):
        raise ValueError("duplicate family names")
    for f in FAMILIES:
        if len(f.templates) < 2:
            raise ValueError(f"{f.name}: needs a dev template and at least one test template")
        for t in f.templates:
            cats = template_categories(t)
            if f.kind == "pos" and not cats:
                raise ValueError(f"{f.name}: positive template without a tagged value: {t[:60]!r}")
            if f.kind == "neg" and cats:
                raise ValueError(f"{f.name}: negative template with a tagged value: {t[:60]!r}")
            for m in PLACEHOLDER.finditer(t):
                if m.group(1) not in PROVIDERS:
                    raise ValueError(f"{f.name}: unknown placeholder {m.group(1)}")
    for split in ("dev", "test"):
        for group, channels in (("W3", W3_CHANNELS), ("W1", W1_S3_CHANNELS)):
            fams = [f for f in FAMILIES if f.channel in channels and group in f.workloads and f.kind == "pos"]
            for cat in CATEGORIES:
                if not any(templates_for(f, split, cat) for f in fams):
                    raise ValueError(f"no {split} {group} template for {cat}")

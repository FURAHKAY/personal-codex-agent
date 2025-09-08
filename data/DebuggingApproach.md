# Debugging Approach

I start small and reproducible. I reduce the failing case to the tiniest snippet or minimal data that still breaks. I read error messages carefully and trace the code path before I touch anything. I test one hypothesis at a time, logging aggressively and writing tiny throwaway scripts. If I get stuck, I step away for five minutes or ask an AI assistant to critique my assumptions. My goal isn’t just to “make it pass” but to understand why it broke so I don’t see the same bug twice.
I approach debugging like detective work:
- **Reproduce first:** I narrow the bug down to a small, repeatable test case.
- **Check assumptions:** I confirm inputs, data types, and edge cases rather than assuming code flow.
- **Divide and conquer:** I isolate modules or lines to locate the failing section.
- **Logs + visualization:** I add targeted printouts or quick plots to “see” what’s happening.
- **Hypothesize & test:** I form small, testable theories instead of random changes.
- **Document the fix:** I leave a clear commit message or comment so others know why it failed.

Mindset: bugs are signals. Each one shows me something I misunderstood — solving them is how I truly *learn the system*.

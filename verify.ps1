Write-Host "🔍 Verifying frontend setup..."

# Check if frontend is still tracked as a submodule
$gitConfig = git config -f .gitmodules --get-regexp submodule.frontend.url 2>$null

if ($gitConfig) {
    Write-Host "⚠️ frontend is still a submodule. Fixing..."

    # Deinit the submodule
    git submodule deinit -f frontend
    # Remove cached index
    git rm -r --cached frontend
    # Remove submodule config
    Remove-Item -Recurse -Force .git\modules\frontend -ErrorAction SilentlyContinue
    git config -f .git/config --remove-section submodule.frontend 2>$null
    git config -f .gitmodules --remove-section submodule.frontend 2>$null

    # Re-add frontend as normal folder
    git add frontend
    git add .gitignore
    git commit -m "Fix: convert frontend from submodule to normal folder"
    git push origin main

    Write-Host "✅ frontend is now a normal folder and changes pushed to GitHub."
} else {
    Write-Host "✅ frontend is already a normal folder. Nothing to fix."
}

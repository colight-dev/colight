import * as assert from 'assert';
import * as vscode from 'vscode';
import * as extension from '../extension';

suite('Extension Test Suite', () => {
    vscode.window.showInformationMessage('Start all tests.');

    test('getCellAtPosition returns correct cell for explicit markers', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: '# %%\nprint("Hello")\n\n# %%\nprint("World")',
            language: 'python'
        });

        // Test first cell
        const cell1 = extension.getCellAtPosition(document, new vscode.Position(1, 0));
        assert.strictEqual(cell1?.text, 'print("Hello")\n');

        // Test second cell
        const cell2 = extension.getCellAtPosition(document, new vscode.Position(4, 0));
        assert.strictEqual(cell2?.text, 'print("World")');
    });

    test('getCellAtPosition returns correct cell for implicit cells', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def func1():\n    print("Hello")\n\ndef func2():\n    print("World")',
            language: 'python'
        });

        // Test first function
        const cell1 = extension.getCellAtPosition(document, new vscode.Position(0, 0));
        assert.strictEqual(cell1?.text, 'def func1():\n    print("Hello")');

        // Test second function
        const cell2 = extension.getCellAtPosition(document, new vscode.Position(3, 0));
        assert.strictEqual(cell2?.text, 'def func2():\n    print("World")');
    });

    test('getCellAtPosition handles empty lines between cells', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def func1():\n    print("Hello")\n\n\ndef func2():\n    print("World")',
            language: 'python'
        });

        // Test first function
        const cell1 = extension.getCellAtPosition(document, new vscode.Position(0, 0));
        assert.strictEqual(cell1?.text, 'def func1():\n    print("Hello")');

        // Test second function
        const cell2 = extension.getCellAtPosition(document, new vscode.Position(4, 0));
        assert.strictEqual(cell2?.text, 'def func2():\n    print("World")');
    });

    test('getCellAtPosition handles nested functions', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def outer():\n    print("Outer")\n    def inner():\n        print("Inner")\n    inner()',
            language: 'python'
        });

        // Test outer function
        const cell = extension.getCellAtPosition(document, new vscode.Position(0, 0));
        assert.strictEqual(cell?.text, 'def outer():\n    print("Outer")\n    def inner():\n        print("Inner")\n    inner()');
    });

    test('getCellAtPosition handles cursor at beginning of def', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def func1():\n    print("Hello")\n\ndef func2():\n    print("World")',
            language: 'python'
        });

        // Test cursor at beginning of first function
        const cell1 = extension.getCellAtPosition(document, new vscode.Position(0, 0));
        assert.strictEqual(cell1?.text, 'def func1():\n    print("Hello")');

        // Test cursor at beginning of second function
        const cell2 = extension.getCellAtPosition(document, new vscode.Position(3, 0));
        assert.strictEqual(cell2?.text, 'def func2():\n    print("World")');
    });

    test('getCellAtPosition handles cursor in middle of nested def', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def outer():\n    print("Outer")\n    def inner():\n        print("Inner")\n    inner()',
            language: 'python'
        });

        // Test cursor in middle of inner function
        const cell = extension.getCellAtPosition(document, new vscode.Position(3, 4));
        assert.strictEqual(cell?.text, 'def outer():\n    print("Outer")\n    def inner():\n        print("Inner")\n    inner()');
    });

    test('getCellAtPosition treats cells separated by empty comment lines as one cell', async () => {
        const document = await vscode.workspace.openTextDocument({
            content: 'def func1():\n    print("Hello")\n#\n    print("Still in func1")\n\ndef func2():\n    print("World")',
            language: 'python'
        });

        // Test first function with empty comment line
        const cell1 = extension.getCellAtPosition(document, new vscode.Position(0, 0));
        assert.strictEqual(cell1?.text, 'def func1():\n    print("Hello")\n#\n    print("Still in func1")');

        // Test second function
        const cell2 = extension.getCellAtPosition(document, new vscode.Position(5, 0));
        assert.strictEqual(cell2?.text, 'def func2():\n    print("World")');
    });

    test('parseMetadata correctly parses cell metadata', () => {
        const metadata = extension.parseMetadata('# %% [markdown] tag="test" id="123"');
        assert.deepEqual(metadata, {type: 'markdown', tag: 'test', id: '123'});
    });

    test('expandSelection grows from cursor to meaningful syntax nodes', async () => {
        // Note: Using raw string to preserve exact indentation
        const text = 
`def greet(name):
    message = "Hello " + name
    print(message)
    return message`;

        const document = await vscode.workspace.openTextDocument({
            content: text,
            language: 'python'
        });
        const editor = await vscode.window.showTextDocument(document);

        // Start with cursor in "message" variable
        editor.selection = new vscode.Selection(1, 4, 1, 4);

        // First expansion: selects "message" variable name
        await vscode.commands.executeCommand('extension.expandSelection');
        let selected = document.getText(editor.selection);
        assert.strictEqual(selected, "message");

        // Second expansion: selects entire assignment
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, 'message = "Hello " + name');

        // Second expansion: selects entire assignment
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, `message = "Hello " + name
    print(message)
    return message`);
        

        // Third expansion: selects entire function body
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, text);
    });

    test('shrinkSelection reverses expansion steps', async () => {
        const text = 
`if True:
    x = 1 + 2
    print(x)`;

        const document = await vscode.workspace.openTextDocument({
            content: text,
            language: 'python'
        });
        const editor = await vscode.window.showTextDocument(document);

        // Start with cursor in the expression
        editor.selection = new vscode.Selection(1, 9, 1, 9);

        // Expand to number
        await vscode.commands.executeCommand('extension.expandSelection');
        let selected = document.getText(editor.selection);
        assert.strictEqual(selected, "1");

        // Expand to full expression
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, "1 + 2");

        // Shrink back to number
        await vscode.commands.executeCommand('extension.shrinkSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, "1");

        // Should return to cursor
        await vscode.commands.executeCommand('extension.shrinkSelection');
        assert.strictEqual(editor.selection.isEmpty, true);
        assert.strictEqual(editor.selection.active.line, 1);
        assert.strictEqual(editor.selection.active.character, 9);
    });

    test('expandSelection handles nested structures correctly', async () => {
        const text = 
`def outer():
    if condition:
        for x in range(10):
            print(x)`;

        const document = await vscode.workspace.openTextDocument({
            content: text,
            language: 'python'
        });
        const editor = await vscode.window.showTextDocument(document);

        // Start with cursor in print statement
        editor.selection = new vscode.Selection(3, 13, 3, 13);

        // First expansion: selects function call
        await vscode.commands.executeCommand('extension.expandSelection');
        let selected = document.getText(editor.selection);
        assert.strictEqual(selected, "print");

        // Second expansion: selects print statement
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, "print(x)");

        // Third expansion: selects for loop block
        await vscode.commands.executeCommand('extension.expandSelection');
        selected = document.getText(editor.selection);
        assert.strictEqual(selected, "for x in range(10):\n            print(x)");
    });

});

